import os
import requests
from typing import Optional,Sequence,Dict,Union,Literal,Any
import base64
from loguru import logger
import time
from concurrent import futures
from dataclasses import dataclass
from tqdm import tqdm
from copy import deepcopy
import json
import pickle
import glob
import shutil
from PIL.Image import Image as PILImage
from io import BytesIO
from itertools import cycle
MULTIMODAL_MODELS = ["gpt-4-vision-preview"]
class GPT:
    def __init__(self,model:str, service:Literal["azure","oai"] = 'azure',api_key:Optional[str] = None,end_point:Optional[str] = None,system_message:Optional[str]=None,detail:Literal["low","high","auto"] = "auto",temperature:float=0.7,top_p:float=0.95,max_tokens:int=800) -> None:
        """A GPT interface.

        Args:
            model (str): The ChatGPT model name. Check https://platform.openai.com/docs/models for more details. Or check Azure OpenAI Studio Model deployment if you use Azure service.
            service (Literal["azure","oai"], optional): The service you use. Set to 'azure' if you use Azure OpenAI service, or 'oai' if you use OpenAI service. Defaults to 'azure'.
            api_key (Optional[str], optional): The API key. If you don't give the api key, we will try to get it from environment variable `GPT_KEY`. Defaults to None.
            end_point (Optional[str], optional): The endpoint. If you don't give the end point, we will try to get it from environment varaible `GPT_ENDPOINT`. Defaults to None.
            system_message (Optional[str], optional): The system message. Currently, we recommend not to use a customized system message, since it seems to make the model refuse to answer some questions. Defaults to None.
            detail (Literal["low","high","auto"], optional): The detail level of the image. This argument is only used when you use a multimodal model. Defaults to "auto".
            temperature (float, optional): GPT temperature. Defaults to 0.7.
            top_p (float, optional): GPT top_p. Defaults to 0.95.
            max_tokens (int, optional): GPT max tokens. Defaults to 800.
            
        """
        api_key = api_key or os.environ.get("GPT_KEY",None)
        assert api_key is not None, "Please set GPT_KEY in environment variable or pass it as an argument."
        end_point = end_point or os.environ.get("GPT_ENDPOINT",None)
        assert end_point is not None, "Please set GPT_ENDPOINT in environment variable or pass it as an argument."
        self.api_key = api_key
        self.end_point = end_point
        self.default_system_message = system_message
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_max_tokens = max_tokens
        self.default_detail = detail
        self.model = model
        self.service = service
        self.__is_multimodal = self.model in MULTIMODAL_MODELS
    def encode_image_to_bytes(self,image:Union[str,bytes,PILImage]) -> str:
        if isinstance(image,str):
            encoded_image = base64.b64encode(open(image, 'rb').read())
        elif isinstance(image,PILImage):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue())
        elif isinstance(image,bytes):
            encoded_image = image
        return encoded_image

    def parse_prompt(self,prompt:str,images: Optional[dict[str,Union[bytes,str,PILImage]]] = None, detail:Literal["low","high","auto"] = "auto") -> list:
        """Parse the prompt, extract and encode image, and finally output the content for GPT.

        Args:
            prompt (str): input prompt. eg. "describe this image <img1>, and this image <img2>"
            images (Optioanl[dict]): a dict of image name and image. The image can be the file path, PILImage or bytes of the image. eg. {"img1": "img1.jpg", "img2": <PILImage>}
            detail (Literal["low","high","auto"], optional): The detail level of the image. Defaults to "auto".
        Returns:
            list: Content list for GPT.
        """
        parsed_prompt = [prompt]
        if isinstance(images,dict):
            for img_name, img_url in images.items():
                new_parsed_prompt = []
                place_holder = f"<{img_name}>"
                for part in parsed_prompt:
                    if isinstance(part,bytes): # this part is an image
                        new_parsed_prompt.append(part)
                        continue
                    else:
                        sub_parts = part.split(place_holder)
                        for idx,sub_part in enumerate(sub_parts):
                            if idx == 0:
                                if sub_part != "":
                                    new_parsed_prompt.append(sub_part)
                                continue
                            img_bytes = self.encode_image_to_bytes(img_url)
                            new_parsed_prompt.append(img_bytes)
                            if sub_part != "":
                                new_parsed_prompt.append(sub_part)
                parsed_prompt = new_parsed_prompt

        content = []
        for part in parsed_prompt:
            if isinstance(part,bytes):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{part.decode('utf-8')}",
                        "detail": detail
                    }
                })
                pass
            else:
                content.append({
                    "type": "text",
                    "text": part
                })

        return content
    def complete(self,prompt:Union[str,list],images: Optional[dict[str,Union[bytes,str,PILImage]]] = None,system_message: Optional[str] = None,detail:Literal["low","high","auto"] = "auto",temperature:Optional[float] = None,top_p: Optional[float] = None, max_tokens: Optional[int] = None,retry:int=-1,debug:bool=False,delay:int=2) -> str:
        """Get the response of a single turn.

        Args:
            prompt (Union[str,list]): input prompt. eg. "describe this image <img1>, and this image <img2>", or a list of turns. eg. [{"role":"user","content":"describe this image <img1>, and this image <img2>"},{"role":"assistant","content":"hello"}]
            images (Optioanl[dict]): a dict of image name and image. The image can be the file path, PILImage or bytes of the image. eg. {"img1": "img1.jpg", "img2": <PILImage>}
            system_message (Optional[str], optional): The system message. Currently, we recommend not to use a customized system message, since it seems to make the model refuse to answer some questions. Defaults to None.
            detail (Literal["low","high","auto"], optional): The detail level of the image. Defaults to "auto".
            temperature (float, optional): GPT temperature. Defaults to 0.7.
            top_p (float, optional): GPT top_p. Defaults to 0.95.
            max_tokens (int, optional): GPT max tokens. Defaults to 800.
            retry (int, optional): The number of retries. Defaults to -1, meaning retring forever.
            debug (bool, optional): Whether to print debug information. Defaults to False.
            delay (int, optional): The delay time between retries. Defaults to 2.
        """
        if not self.__is_multimodal and images is not None:
            logger.warning(f"You're using a single-modal model {self.model}, but you give images. The images will be ignored.")

        detail = detail or self.default_detail

        if self.service == "oai":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
    
        # Payload for the request
        messages = []
        sm = system_message or self.default_system_message
        if sm is not None:
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message
                    }
                ] if self.__is_multimodal else system_message
            })
        if isinstance(prompt,str):
            messages.append({
                "role": "user",
                "content": self.parse_prompt(prompt,images,detail) if self.__is_multimodal else prompt
            })
        elif isinstance(prompt,list):
            # Given system message overwrites the default system message
            if prompt[0]['role'] == 'system':
                messages = []
            for turn in prompt:
                role = turn['role']
                content = turn['content']
                content = self.parse_prompt(content,images,detail) if self.__is_multimodal else content
                messages.append({
                    "role": role,
                    "content": content
                })
        payload = {
          "messages": messages,
          "temperature": temperature if temperature is not None else self.default_temperature,
          "top_p": top_p if top_p is not None else self.default_top_p,
          "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
          "model": self.model
        }

        # Send request
        while retry!=0:
            try:
                response = requests.post(self.end_point, headers=headers, json=payload)
                response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                response = response.json()
                return response['choices'][0]['message']['content']
            except requests.RequestException as e:
                if debug:
                    logger.exception(e)
                if 'Too Many Requests' in str(e):
                    time.sleep(delay)
                else:
                    return None
            retry -= 1
        return -1

@dataclass
class GPTFactoryOutput:
    id:Any = None
    input:dict = None
    response:str = None
    job_id: int = None

class GPTFactory:
    def __init__(self,keys_endpoints:Sequence[Dict[str,str]],model:str,service:Literal["azure","oai"]='azure',system_message:Optional[str]=None,detail:Literal["low","high","auto"] = "auto" ,temperature:float=0.7,top_p:float=0.95,max_tokens:int=800,debug=False) -> None:
        """A Factory containing multiple GPT chatbot, which can process multiple jobs at the same time.
        
        Args:
            keys_endpoints (Sequence[Dict[str,str]]): A list of dict, each dict contains the api key and endpoint. eg. [{"api_key":"xxx","end_point":"xxx"},{"api_key":"xxx","end_point":"xxx"}]
            model (str): The ChatGPT model name. Check https://platform.openai.com/docs/models for more details. Or check Azure OpenAI Studio Model deployment if you use Azure service.
            service (Literal["azure","oai"], optional): The service you use. Set to 'azure' if you use Azure OpenAI service, or 'oai' if you use OpenAI service. Defaults to 'azure'.
            system_message (Optional[str], optional): The system message. Currently, we recommend not to use a customized system message, since it seems to make the model refuse to answer some questions. Defaults to None.
            detail (Literal["low","high","auto"], optional): The detail level of the image. Defaults to "auto".
            temperature (float, optional): GPT temperature. Defaults to 0.7.
            top_p (float, optional): GPT top_p. Defaults to 0.95.
            max_tokens (int, optional): GPT max tokens. Defaults to 800.

        """
        self.keys_endpoints = keys_endpoints
        self.default_system_message = system_message
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_max_tokens = max_tokens
        self.default_detail = detail
        self.model = model
        self.service = service
        self.__started = False
        self.debug = debug
        self.__init_chatbots()
    
    def set_default_value(self,name:Literal["system_message","temperature","top_p","max_tokens","detail"],value:Any) -> None:
        assert name in ["system_message","temperature","top_p","max_tokens","detail"], f"The attribute {name} does not exist!"
        if name == "system_message":
            assert isinstance(value,str), "The system_message must be a string!"
        elif name == "temperature":
            assert isinstance(value,float), "The temperature must be a float!"
        elif name == "top_p":
            assert isinstance(value,float), "The top_p must be a float!"
        elif name == "max_tokens":
            assert isinstance(value,int), "The max_tokens must be an int!"
        elif name == "detail":
            assert value in ["low","high","auto"], "The detail must be one of 'low','high','auto'!"
        name = f"default_{name}"
        setattr(self,name,value)

    def __worker(self,args):
        id = args.pop('id',None)
        job_id = args.pop('job_id',None)
        for chatbot in cycle(self.chatbots):
            response = chatbot.complete(**args)
            if response == -1:
                continue
            else:
                break
        if id is not None:
            args['id'] = id
        output = GPTFactoryOutput(id,args,response,job_id)
        if job_id is not None:
            args['job_id'] = job_id
        return output


    def __init_chatbots(self):
        self.chatbots = []
        for key_endpoint in self.keys_endpoints:
            api_key = key_endpoint['api_key']
            end_point = key_endpoint['end_point']
            chatbot = GPT(self.model,self.service,api_key,end_point,self.default_system_message,self.default_detail,self.default_temperature,self.default_top_p,self.default_max_tokens)
            self.chatbots.append(chatbot)
            
    
    def start(self) -> None:
        """Start all chatbot threads. This should be run before you put jobs.
        """
        if self.__started:
            logger.warning("You start the factory again. But the GPTFactory has already started.")
            return

        self.__started = True
        self.executor = futures.ThreadPoolExecutor(max_workers=len(self.chatbots))
    
    def stop(self) -> None:
        """Stop all chatbot threads. This should be run after you get all results.
        """
        if not self.__started:
            logger.warning("You stop the factory again. But the GPTFactory has already stopped.")
            return
        self.__started = False
        self.executor.shutdown()
    
    def restart(self) -> None:
        """Restart all chatbot threads.
        """
        self.stop()
        self.start()

    def __put(self,args:dict):
        future = self.executor.submit(self.__worker,args)
        return future

    def put(self,prompt:Union[str,list],images: Optional[dict[str,Union[bytes,str,PILImage]]], system_message: Optional[str] = None,detail:Literal["low","high","auto"] = "auto",temperature:Optional[float] = None,top_p: Optional[float] = None, max_tokens: Optional[int] = None, id: Optional[Any] = None,**kwargs) -> None:
        """Put a job into the job queue. You can get the result by calling `get()`.

        Args:
            prompt (str): input prompt. eg. "describe this image <img1>, and this image <img2>", or a list of turns. eg. [{"role":"user","content":"describe this image <img1>, and this image <img2>"},{"role":"assistant","content":"hello"}]
            images (Optioanl[dict]): a dict of image name and image. The image can be the file path, PILImage or bytes of the image. eg. {"img1": "img1.jpg", "img2": <PILImage>}
            system_message (Optional[str], optional): The system message. Currently, we recommend not to use a customized system message, since it seems to make the model refuse to answer some questions. Defaults to None.
            detail (Literal["low","high","auto"], optional): The detail level of the image. Defaults to "auto".
            temperature (float, optional): GPT temperature. Defaults to 0.7.
            top_p (float, optional): GPT top_p. Defaults to 0.95.
            max_tokens (int, optional): GPT max tokens. Defaults to 800.
            id (Optional[Any], optional): The id of the job. Defaults to None.
        """
        args = {
            "prompt":prompt,
            "images":images,
            "system_message":system_message,
            "detail":detail,
            "temperature":temperature,
            "top_p":top_p,
            "max_tokens":max_tokens,
            "id":id,
            "retry":1,
            "delay":0.2,
            "debug":self.debug
        }
        other_args = deepcopy(kwargs)
        args.update(other_args)
        return self.__put(args)

    def __save(self,inputs,results,error_jobs,save_path:str,cur_step:int,save_total_limit:int=1):
        checkpoint_num = len(glob.glob(os.path.join(save_path,"checkpoint_*")))
        if checkpoint_num >= save_total_limit:
            checkpoint_name_sorted_by_step = sorted(glob.glob(os.path.join(save_path,"checkpoint_*")),key=lambda x:int(x.split("_")[-1]))
            checkpoint_name_to_delete = checkpoint_name_sorted_by_step[0]
            shutil.rmtree(checkpoint_name_to_delete)
        save_path = os.path.join(save_path,f"checkpoint_{cur_step}")
        os.makedirs(save_path,exist_ok=True)
        inputs_file = os.path.join(save_path,"inputs.pkl")
        
        with open(inputs_file,"wb") as f:
            pickle.dump(inputs,f)
        with open(os.path.join(save_path,"error_jobs.pkl"),"wb") as f:
            pickle.dump(error_jobs,f)
        results_file = os.path.join(save_path,"results.pkl")
        with open(results_file,"wb") as f:
            pickle.dump(results,f)
    
    def load_checkpoint(self,checkpoint_path:str) -> tuple[Sequence, list, list]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path (str): The path of the checkpoint.
        Returns:
            tuple[Sequence, list, list]: A tuple of inputs, results and error_jobs.
        """
        assert os.path.isdir(checkpoint_path), f"The checkpoint_path {checkpoint_path} is not a directory!"
        inputs_file = os.path.join(checkpoint_path,"inputs.pkl")
        if not os.path.isfile(inputs_file):
            logger.exception(f"No inputs file found in {checkpoint_path}!")
            raise FileNotFoundError
        else:
            with open(inputs_file,"rb") as f:
                inputs = pickle.load(f)
        try:
            results_file = glob.glob(os.path.join(checkpoint_path,"results.*"))[0]
            if results_file.endswith(".json"):
                with open(results_file,"r") as f:
                    results = json.load(f)
            elif results_file.endswith(".pkl"):
                with open(results_file,"rb") as f:
                    results = pickle.load(f)
        except IndexError:
            logger.warning(f"No results file found in {checkpoint_path}!")
            results = []

        error_file = os.path.join(checkpoint_path,"error_jobs.pkl")
        if os.path.isfile(error_file):
            with open(error_file,"rb") as f:
                error_jobs = pickle.load(f)
        else:
            error_jobs = []

        return inputs,results,error_jobs

    def run_task(self, inputs: Optional[Sequence[dict]]=None,resume_from_checkpoint:Optional[str] = None,rerun_error_jobs:bool = False,save_path:Optional[str] = None,save_step:Optional[int] = None,save_total_limit:int=1,show_progress=True) -> list[GPTFactoryOutput]:
        """Run a bunch of jobs.

        Args:
            inputs (Optional[Sequence[dict]], optional): A list of inputs. Each input is a dict. eg. [{"prompt":prompt,"images":images,"id":id,"temperature":0.7},...]. Defaults to None. Hyperparameters in the input will overwrite the default hyperparameters.
            resume_from_checkpoint (Optional[str], optional): The path of the checkpoint. Defaults to None.
            rerun_error_jobs (bool, optional): Whether to rerun the error jobs. Defaults to False.
            save_path (Optional[str], optional): The path to save the checkpoint. Defaults to None.
            save_step (Optional[int], optional): The step to save the checkpoint. Defaults to None.
            save_total_limit (int, optional): The maximum number of checkpoints to save. Defaults to 1.
            show_progress (bool, optional): Whether to show the progress bar. Defaults to True.
        Returns:
            list[GPTFactoryOutput]: A list of outputs. Each output is a GPTFactoryOutput. You can get the response by `output.response`.
        """
        if inputs is not None and resume_from_checkpoint is not None:
            logger.exception("You give both inputs and resume_from_checkpoint. Please only give one of them!")
            raise ValueError
        if inputs is None and resume_from_checkpoint is None:
            logger.exception("You give neither inputs nor resume_from_checkpoint. Please give one of them!")
            raise ValueError
        futures = []
        if self.__started:
            logger.warning("The factory has already started. We will empty the job queue and result queue, and restart the factory!")
            self.stop()
        self.start()
        if resume_from_checkpoint is not None:
            inputs,results,error_jobs = self.load_checkpoint(resume_from_checkpoint)
            logger.info(f"Resume from checkpoint {resume_from_checkpoint}.")
            if rerun_error_jobs:
                for error_job in error_jobs:
                    job_id = error_job['job_id']
                    inputs[job_id].pop('complete')
                results_without_error_jobs = []
                for result in results:
                    if result.response is not None:
                        results_without_error_jobs.append(result)
                results = results_without_error_jobs
                error_jobs = []
            for input in inputs:
                if 'complete' not in input:
                    futures.append(self.put(**input))

        else:
            results = []
            error_jobs = []
            inputs = deepcopy(inputs)
            logger.info(f"Start processing {len(inputs)} inputs.")
            for idx,input in enumerate(inputs):
                input['job_id'] = idx # this job_id is used to identify the raw input of the output
                futures.append(self.put(**input))
        
        bar_format = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}, {desc}]"
        if show_progress:
            progress_bar = tqdm(total=len(inputs),initial=len(results),bar_format=bar_format,desc=f"{len(error_jobs)} errors",dynamic_ncols=True)
        step = len(results)
        for future in futures:
            output = future.result()
            job_id = output.job_id
            inputs[job_id]['complete'] = True
            if output.response is None:
                error_jobs.append(inputs[job_id])
            results.append(output)
            if show_progress:
                progress_bar.update(1)
                progress_bar.set_description_str(f"{len(error_jobs)} errors")
            step += 1
            if save_path is not None and save_step is not None and step % save_step == 0:
                self.__save(inputs,results,error_jobs,save_path,step,save_total_limit)
        self.stop()
        if show_progress:
            progress_bar.close()
        logger.success(f"Successfully processing {len(results)-len(error_jobs)} inputs. {len(error_jobs)} errors. Factory stopped.")
        if save_path is not None:
            self.__save(inputs,results,error_jobs,save_path,step,save_total_limit)
        return results


def smart_build_factory(api_info:Sequence[Dict],model:str,service:Literal["azure","oai"]='azure',worker_num:Optional[int] = None,system_message:Optional[str]=None,detail:Literal["low","high","auto"] = "auto",temperature:float=0.7,top_p:float=0.95,max_tokens:int=800,debug:bool=False) -> GPTFactory:
    """Build a GPTFactory with any number of worker. The api each worker used is smartly assigned, to balance the workload.

    Args:
        api_info (Sequence[Dict]): A list of dict, each dict contains the api key and endpoint. eg. [{"api_key":"xxx","end_point":"xxx"},{"api_key":"xxx","end_point":"xxx"}]
        model (str): The ChatGPT model name. Check https://platform.openai.com/docs/models for more details. Or check Azure OpenAI Studio Model deployment if you use Azure service.
        service (Literal["azure","oai"], optional): The service you use. Set to 'azure' if you use Azure OpenAI service, or 'oai' if you use OpenAI service. Defaults to 'azure'.
        worker_num (Optional[int], optional): The number of workers. If None, we will use the number of api_info. Defaults to None.
        detail (Literal["low","high","auto"], optional): The detail level of the image. Defaults to "auto".
        temperature (float, optional): GPT temperature. Defaults to 0.7.
        top_p (float, optional): GPT top_p. Defaults to 0.95.
        max_tokens (int, optional): GPT max tokens. Defaults to 800.
        system_message (Optional[str], optional): The system message. Currently, we recommend not to use a customized system message, since it seems to make the model refuse to answer some questions. Defaults to None.
        debug (bool, optional): Whether to print debug information. Defaults to False.
    Returns:
        GPTFactory: A GPTFactory.
    """
    if worker_num is None:
        worker_num = len(api_info)
    if worker_num % len(api_info) != 0:
        logger.warning(f"The worker_num {worker_num} is not a multiple of the number of api_info {len(api_info)}. The workload may not be balanced.")

    keys_endpoints = []
    for i in range(worker_num):
        keys_endpoints.append(api_info[i%len(api_info)])
    return GPTFactory(keys_endpoints,model,service,system_message,detail,temperature,top_p,max_tokens,debug)