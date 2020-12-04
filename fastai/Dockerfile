FROM public.ecr.aws/lambda/python:3.7

RUN python3.7 -m pip install --no-cache-dir -t . \
    torch==1.7.0+cpu torchvision==0.8.1+cpu fastai \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY app.py export.pkl  ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]