import json
import time
from io import BytesIO

import requests
import numpy as np
from fastai.vision.all import load_learner, PILImage

learn = load_learner('export.pkl')

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    #print("Received event: " + json.dumps(event, indent=2))
    body = json.loads(event['body'])
    print(f"Body is: {body}")
    url = body['url']
    print(f"Getting image from URL: {url}")
    response = requests.get(url)
    print("Load image into memory")
    img = PILImage.create(BytesIO(response.content))
    print("Doing forward pass")
    start = time.time()
    pred,pred_idx,probs = learn.predict(img)
    end = time.time()
    inference_time = np.round((end - start) * 1000, 2)
    print(f'class: {pred}, probability: {probs[pred_idx]:.04f}')
    print(f'Inference time is: {str(inference_time)} ms')
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "class": pred,
                "probability": "%.4f" % probs[pred_idx]
            }
        ),
    }

