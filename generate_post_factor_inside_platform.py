from utils.database import DataBase

db = DataBase('data/filtered_posts.csv', 'data/all_accounts.csv')
db.cal_post_factor_inside_platform()