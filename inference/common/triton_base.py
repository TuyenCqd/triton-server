import sys
import time
import numpy as np
import glog as logger
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

class TritonBaseClient:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = 'localhost:8001', # default gRPC port
                triton_model_name = '',
                connection = 'GRPC',
                verbose = False,
                ssl = False,
                root_certificates = None,
                private_key = None,
                certificate_chain = None,
                max_batch_size = 1, 
                shared_memory = False,
                shared_cuda_memory = False):
        
        assert connection in ['GRPC', 'HTTP'], "Current support only connection type GRPC or HTTP"
        logger.info('Init connection from Triton-inference-server')
        logger.info('- Host: {}'.format(triton_host))
        logger.info('- Model: {}'.format(triton_model_name))
        logger.info('- Connection: {}'.format(connection))
        logger.info('- Shared memory: {}'.format(shared_memory))

        self.triton_host = triton_host
        self.triton_model_name = triton_model_name
        self.connection = connection
        if self.connection == 'GRPC':
            self.model = grpcclient.InferenceServerClient(url = self.triton_host,
                                                        verbose = verbose,
                                                        ssl = ssl,
                                                        root_certificates = root_certificates,
                                                        private_key = private_key,
                                                        certificate_chain = certificate_chain)
        else:
            self.model = httpclient.InferenceServerClient(url = self.triton_host)
        if not self.model.is_server_live():
            logger.info("[ERROR] Server not found: {}".format(self.triton_host))
            sys.exit(1)
        
        if not self.model.is_model_ready(self.triton_model_name):
            logger.info("[ERROR] Model not ready: {}".format(self.triton_model_name))
            sys.exit(1)
        
        self.max_batch_size = max_batch_size
        self.shared_memory = shared_memory
        self.shared_cuda_memory = shared_cuda_memory

    def preprocess(self, imgs):
        """
            Preprocess image
            Input: List of image
            Output: Batch image normalization
        """
        pass
    
    def postprocess(self, batch_result):
        pass
        
    def run(self, batch_data, meta_inputs, meta_outputs, verbose = False):
        
        if verbose:
            tik = time.time()
        if isinstance(batch_data, list):
            total_images = len(batch_data[0]) 
        else:
            total_images = len(batch_data)

        total_batchs = int(total_images/self.max_batch_size) if total_images % self.max_batch_size == 0 else int(total_images/self.max_batch_size) + 1
        batch_results = []
        
        for ib in range(total_batchs):
            inputs = []
            outputs = []
            lower = ib * self.max_batch_size
            higher = min((ib+1)*self.max_batch_size, total_images)
            # if verbose:
            #     logger.info(' --> Infer batch {} from data range {}-{}'.format(ib, lower, higher))
            if isinstance(batch_data, list) and len(batch_data) == len(meta_inputs):
                data = batch_data
            else:
                data = [batch_data[lower:higher]]
            if self.connection == 'GRPC':
                for ix, input_tuple in enumerate(meta_inputs):
                    inputs.append(grpcclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
                    inputs[ix].set_data_from_numpy(data[ix])
                    
                for ix, output_tuple in enumerate(meta_outputs):
                    outputs.append(grpcclient.InferRequestedOutput(output_tuple[0]))
            else:
                for ix, input_tuple in enumerate(meta_inputs):
                    inputs.append(httpclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
                    inputs[ix].set_data_from_numpy(data[ix])

                for ix, output_tuple in enumerate(meta_outputs):
                    outputs.append(httpclient.InferRequestedOutput(output_tuple[0]))

            results = self.model.infer(
                model_name=self.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=None)
                
            results_dict = {}
            for ix, output_tuple in enumerate(meta_outputs):
                output_np = results.as_numpy(output_tuple[0])
                results_dict[output_tuple[0]] = output_np.copy()

            for i in range(higher - lower):
                result_per_image = {}
                for ix, output_tuple in enumerate(meta_outputs):
                    output_name = output_tuple[0]
                    data_from_server = results_dict[output_name]

                    if data_from_server.size > 0:
                        result_per_image[output_name] = data_from_server[i]
                    else:
                        result_per_image[output_name] = np.array([]) 
                    # print('output_tuple: ', output_tuple)
                    # result_per_image[output_tuple[0]] = results_dict[output_tuple[0]][i]
                    # # print('result_per_image: ',result_per_image)
                batch_results.append(result_per_image)

        if verbose:
            time_infer = time.time() - tik
            logger.info(f'[INFO] Inference cost: {int(time_infer * 1000)}ms')
        
        return batch_results