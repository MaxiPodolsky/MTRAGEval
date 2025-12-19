import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_KEY = "sk-7694bfac92304ac38a682d437a382358"
#MILVUS_URI = 'http://1.95.116.112:19530'
MILVUS_URI = '127.0.0.1:19530'

#COLLECTION_NAME = 't_collection01'  #暂且就叫这个名字，后面再创新的时候再改
COLLECTION_NAME = 'clapnq_1000'  #先拿1000的试一试