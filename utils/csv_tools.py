import pandas as pd

def read_csv_post(file_path):
    """
    读取csv文件
    :param file_path:
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path, encoding='utf-8', dtype={'user_id': str, 'post_id': str})
    df = df.where(df.notnull(), None)
    return df


def read_csv_user(file_path):
    """
    读取csv文件
    :param file_path:
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path, encoding='utf-8', dtype={'user_id': str})
    df = df.where(df.notnull(), None)
    return df