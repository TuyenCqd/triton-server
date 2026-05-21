#!/usr/bin/env python3
"""
Performance benchmarking script for Face Alignment backend
Compares Python backend vs C++ GPU backend
"""

import tritonclient.grpc as grpcclient
import time
import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from statistics import mean, stdev

class FaceAlignmentBenchmark:
    def __init__(self, server_url="localhost:8001", model_name="face_alignment"):
        self.client = grpcclient.InferenceServerClient(server_url)
        self.model_name = model_name
        
    def prepare_test_data(self, img_height=1080, img_width=1920, batch_size=1):
        """Generate synthetic test data"""
        # Random BGR image
        image = np.random.randint(0, 255, (img_height, img_width, 3), dtype=np.uint8)
        
        # Standard landmarks (5 facial keypoints)
        landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Bounding box [x1, y1, x2, y2]
        bbox = np.array([[100, 100, 600, 700]], dtype=np.float32)
        
        # Stack for batch
        images = np.repeat(image[np.newaxis], batch_size, axis=0)
        landmarks_batch = np.repeat(landmarks[np.newaxis], batch_size, axis=0)
        bboxes = np.repeat(bbox, batch_size, axis=0)
        
        return images, landmarks_batch, bboxes
    
    def run_inference(self, images, landmarks, bboxes, num_runs=10, warmup=2):
        """Run inference and measure latency"""
        # Warmup runs
        for _ in range(warmup):
            self._infer_once(images, landmarks, bboxes)
        
        # Measurement runs
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = self._infer_once(images, landmarks, bboxes)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # Convert to ms
        
        return result, latencies
    
    def _infer_once(self, images, landmarks, bboxes):
        """Single inference call"""
        inputs = [
            grpcclient.InferInput("person_image", images.shape, "UINT8"),
            grpcclient.InferInput("landmarks", landmarks.shape, "FP32"),
            grpcclient.InferInput("bboxes", bboxes.shape, "FP32"),
        ]
        
        inputs[0].set_data_from_numpy(images)
        inputs[1].set_data_from_numpy(landmarks)
        inputs[2].set_data_from_numpy(bboxes)
        
        outputs = [
            grpcclient.InferRequestedOutput("face_aligned_112"),
            grpcclient.InferRequestedOutput("face_aligned_224"),
            grpcclient.InferRequestedOutput("face_aligned_nhwc"),
        ]
        
        response = self.client.infer(
            self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        return {
            "face_aligned_112": response.as_numpy("face_aligned_112"),
            "face_aligned_224": response.as_numpy("face_aligned_224"),
            "face_aligned_nhwc": response.as_numpy("face_aligned_nhwc"),
        }
    
    def print_stats(self, name, latencies):
        """Print latency statistics"""
        latencies_sorted = sorted(latencies)
        
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")
        print(f"Min Latency:     {min(latencies_sorted):.2f} ms")
        print(f"Max Latency:     {max(latencies_sorted):.2f} ms")
        print(f"Mean Latency:    {mean(latencies):.2f} ms")
        print(f"Median Latency:  {latencies_sorted[len(latencies_sorted)//2]:.2f} ms")
        
        if len(latencies) > 1:
            print(f"Stdev Latency:   {stdev(latencies):.2f} ms")
        
        print(f"p50 Latency:     {latencies_sorted[int(len(latencies)*0.50)]:.2f} ms")
        print(f"p95 Latency:     {latencies_sorted[int(len(latencies)*0.95)]:.2f} ms")
        print(f"p99 Latency:     {latencies_sorted[int(len(latencies)*0.99)]:.2f} ms")
        
        throughput = (1000.0 / mean(latencies)) if mean(latencies) > 0 else 0
        print(f"Throughput:      {throughput:.2f} inferences/sec")
        
        return {
            "min": min(latencies_sorted),
            "max": max(latencies_sorted),
            "mean": mean(latencies),
            "median": latencies_sorted[len(latencies_sorted)//2],
            "stdev": stdev(latencies) if len(latencies) > 1 else 0,
            "p50": latencies_sorted[int(len(latencies)*0.50)],
            "p95": latencies_sorted[int(len(latencies)*0.95)],
            "p99": latencies_sorted[int(len(latencies)*0.99)],
            "throughput": throughput,
        }
    
    def benchmark_batch_sizes(self, batch_sizes=[1, 4, 8, 16, 32], num_runs=20):
        """Benchmark different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"Testing Batch Size: {batch_size}")
            print(f"{'='*60}")
            
            images, landmarks, bboxes = self.prepare_test_data(batch_size=batch_size)
            result, latencies = self.run_inference(images, landmarks, bboxes, num_runs=num_runs)
            
            stats = self.print_stats(f"Batch Size {batch_size}", latencies)
            results[f"batch_{batch_size}"] = stats
            
            # Verify outputs
            print(f"\nOutput Shapes:")
            print(f"  face_aligned_112: {result['face_aligned_112'].shape}")
            print(f"  face_aligned_224: {result['face_aligned_224'].shape}")
            print(f"  face_aligned_nhwc: {result['face_aligned_nhwc'].shape}")
        
        return results
    
    def benchmark_concurrency(self, concurrency_levels=[1, 2, 4, 8], num_runs=20):
        """Benchmark concurrent requests (simulated)"""
        import threading
        import queue
        
        results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n{'='*60}")
            print(f"Testing Concurrency Level: {concurrency}")
            print(f"{'='*60}")
            
            latencies = []
            q = queue.Queue()
            
            def worker():
                images, landmarks, bboxes = self.prepare_test_data(batch_size=1)
                for _ in range(num_runs):
                    start = time.perf_counter()
                    self._infer_once(images, landmarks, bboxes)
                    elapsed = time.perf_counter() - start
                    q.put(elapsed * 1000)
            
            threads = [threading.Thread(target=worker) for _ in range(concurrency)]
            
            start_time = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            total_time = time.perf_counter() - start_time
            
            # Collect results
            while not q.empty():
                latencies.append(q.get())
            
            stats = self.print_stats(f"Concurrency {concurrency}", latencies)
            stats["total_time"] = total_time
            stats["requests"] = concurrency * num_runs
            results[f"concurrency_{concurrency}"] = stats
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Face Alignment Backend Benchmark")
    parser.add_argument("--server", default="localhost:8001", help="Triton server address")
    parser.add_argument("--model", default="face_alignment", help="Model name")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8],
                       help="Concurrency levels to test")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"Connecting to Triton server at {args.server}")
    benchmark = FaceAlignmentBenchmark(args.server, args.model)
    
    # Check model status
    print(f"Checking model status...")
    model_metadata = benchmark.client.get_model_metadata(args.model)
    print(f"Model loaded: {args.model}")
    
    # Run benchmarks
    print(f"\n{'='*60}")
    print(f"BATCH SIZE BENCHMARKS")
    print(f"{'='*60}")
    batch_results = benchmark.benchmark_batch_sizes(args.batch_sizes, args.runs)
    
    print(f"\n{'='*60}")
    print(f"CONCURRENCY BENCHMARKS")
    print(f"{'='*60}")
    concurrency_results = benchmark.benchmark_concurrency(args.concurrency, args.runs)
    
    # Save results
    all_results = {
        "batch_sizes": batch_results,
        "concurrency": concurrency_results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
