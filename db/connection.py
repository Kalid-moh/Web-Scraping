import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host= os.getenv("host"),
        port= os.getenv("port"),
        user= os.getenv("user"),          
        password= os.getenv("password"),           
        database= os.getenv("database")    
    )
