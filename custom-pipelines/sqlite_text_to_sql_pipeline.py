import sqlite3
import logging
from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from sqlalchemy import create_engine
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
import aiohttp
import asyncio



logging.basicConfig(level=logging.DEBUG)

class Pipeline:
    class Valves(BaseModel):        
        DB_TABLES: List[str]
        OPENAI_URL: str
        OPENAI_API_KEY: str
        OPENAI_MODEL: str

    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.cur = None
        self.conn = None
        self.nlsql_response = ""
        
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],                
                "DB_TABLES": ["channels", "subscriber_counts"],
                "OPENAI_URL": os.getenv("OPENAI_URL", "https://openrouter.ai/api/v1"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", 'abc-123'),
                "OPENAI_MODEL": os.getenv("OPENAI_API_KEY", 'google/gemini-2.0-flash-001')
            }
        )

    def init_db_connection(self):
        
        db_path = 'fowl.db'
        try:
            # Open connection to the database
            self.conn = sqlite3.connect(db_path)
            print(f"Successfully connected to SQLite database: {db_path}")
        except sqlite3.Error as error:
            print(f"Error connecting to the Sqlite database: {error}")            

        # Create a cursor object
        self.cur = self.conn.cursor()

        # Query to get the list of tables
        # Fetch and print the table names
        self.cur.execute(" SELECT name FROM sqlite_master WHERE type='table'; ")
        tables = self.cur.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(f"{table}")

        self.cur.close()
        self.conn.close()
        
    async def on_startup(self):
        self.init_db_connection()

    async def on_shutdown(self):
        self.cur.close()
        self.conn.close()

    async def make_request_with_retry(self, url, params, retries=3, timeout=10):
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=timeout) as response:
                        response.raise_for_status()
                        return await response.text()
            except (aiohttp.ClientResponseError, aiohttp.ClientPayloadError, aiohttp.ClientConnectionError) as e:
                logging.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def extract_sql_query(self, response_object):
        for key, value in response_object.items():
            if isinstance(value, dict) and 'sql_query' in value:
                return value['sql_query']
            elif key == 'sql_query':
                return value
        return None

    def handle_streaming_response(self, response_gen):
        final_response = ""
        for chunk in response_gen:
            final_response += chunk
        return final_response

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # create a SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.valves.DB_PATH}")
        sql_database = SQLDatabase(self.engine, include_tables=self.valves.DB_TABLES)

        llm = OpenAILike(
            model=self.valves.OPENAI_MODEL,
            api_base=self.valves.OPENAI_URL,
            api_key=self.valves.OPENAI_API_KEY,
            max_tokens=100,
        )
        
        text_to_sql_prompt = """
        You are a helpful AI Assistant providing Sqlite3 commands to users.
        Given an input question, create a syntactically correct Sqlite3 query to run.
        
        Only use tables listed below.
        
        CREATE TABLE channels
        (channel_id TEXT PRIMARY KEY, channel_name TEXT);
        
        CREATE TABLE subscriber_counts
        (timestamp TEXT,
        channel_id TEXT,
        subscriber_count INTEGER,
        FOREIGN KEY(channel_id) REFERENCES channels(channel_id));
        
        Question: {query_str}
        """
        
        synthesis_prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>    
        You are a helpful AI Assistant synthesizing the response from a Sqlite3 query.
        
        SQLResponse: 
        
        
        
        ******** edit this *********
        
        
        
        Only use tables listed below.
        movies
        
        Only use columns listed below.
        [('Release Year',), ('title',), ('Origin/Ethnicity',), ('director',), ('Cast',), ('genre',), ('Wiki Page',), ('plot',)]
        
        Question: How many rows in the database?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        SQLQuery: SELECT COUNT(*) FROM "movies"<|eot_id|>
        
        <|start_header_id|>user<|end_header_id|>
        Only use tables listed below.
        movies
        
        Only use columns listed below.
        [('Release Year',), ('title',), ('Origin/Ethnicity',), ('director',), ('Cast',), ('genre',), ('Wiki Page',), ('plot',)]
        
        Question: How many comedy movies in 1995?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        SQLQuery: SELECT COUNT(*) FROM "movies" WHERE "Release Year" = 1995 AND "genre" = 'comedy'<|start_header_id|>user<|end_header_id|>
        
        Only use tables listed below.
        movies
        
        Only use columns listed below.
        [('Release Year',), ('title',), ('Origin/Ethnicity',), ('director',), ('Cast',), ('genre',), ('Wiki Page',), ('plot',)]
        
        Question: {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        text_to_sql_template = PromptTemplate(text_to_sql_prompt)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=self.valves.DB_TABLES,
            llm=llm,
            embed_model="local",
            text_to_sql_prompt=text_to_sql_template,
            synthesize_response=False,
            streaming=True
        )

        try:
            response = query_engine.query(user_message)
            sql_query = self.extract_sql_query(response.metadata)
            if hasattr(response, 'response_gen'):
                final_response = self.handle_streaming_response(response.response_gen)
                result = f"Generated SQL Query:\n```sql\n{sql_query}\n```\nResponse:\n{final_response}"
                self.engine.dispose()
                return result
            else:
                final_response = response.response
                result = f"Generated SQL Query:\n```sql\n{sql_query}\n```\nResponse:\n{final_response}"
                self.engine.dispose()
                return result
        except aiohttp.ClientResponseError as e:
            logging.error(f"ClientResponseError: {e}")
            self.engine.dispose()
            return f"ClientResponseError: {e}"
        except aiohttp.ClientPayloadError as e:
            logging.error(f"ClientPayloadError: {e}")
            self.engine.dispose()
            return f"ClientPayloadError: {e}"
        except aiohttp.ClientConnectionError as e:
            logging.error(f"ClientConnectionError: {e}")
            self.engine.dispose()
            return f"ClientConnectionError: {e}"
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            self.engine.dispose()
            return f"Unexpected error: {e}"
