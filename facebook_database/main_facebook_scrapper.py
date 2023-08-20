# https://github.com/kevinzg/facebook-scraper


from facebook_scraper import get_posts
for post in get_posts('nintendo', pages=3):
    print(post['text'][:50])