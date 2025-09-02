FROM whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2

# install python dependencies
RUN pip install loguru tqdm==4.66.0 ninja tensorboard nvitop rich \
        sentencepiece \
        transformers==4.51.2 \
        torchvision==0.20.1 hydra-core accelerate \
        ray==2.40.0 deepspeed==0.16.0 vllm==0.6.5 peft datasets openpyxl swanlab protobuf==3.20.0 math-verify

RUN pip install pynvml --upgrade

# set time zone to CST
RUN apt update && apt install -y tzdata  \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt clean