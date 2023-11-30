from dbutils.pooled_db import PooledDB
import pymysql
import json

class MySQLInserter:
    def __init__(self, host, user, password, db_name):
        self.pool = PooledDB(
            creator=pymysql,  # 使用pymysql连接数据库
            maxconnections=10,  # 连接池允许的最大连接数
            host=host,
            user=user,
            password=password,
            database=db_name
        )

    def insert_data(self, table_name: str, data: json):
        """插入数据到指定表"""
        connection = self.pool.connection()
        cursor = connection.cursor()

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        values = tuple(data.values())

        try:
            cursor.execute(sql, values)
            connection.commit()
        except Exception as e:
            print(f"Error: {e}")
            connection.rollback()
        finally:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    inserter = MySQLInserter('localhost', 'vision', 'Vision@123456.com', 'django-vue-admin')
    json_data_str = '{"update_datetime": "2023-10-11T12:00:00", "create_datetime": "2023-10-11T12:00:00", "camera_name": "Camera1", "model_name": "ModelA", "detection_time": "2023-10-11T12:05:00", "image_save_path": "/path/to/image.jpg", "result": "Success"}'
    inserter.insert_data('test', json.loads(json_data_str))
