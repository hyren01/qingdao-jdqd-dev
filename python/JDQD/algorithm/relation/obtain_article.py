import psycopg2
import time
import re
import os
import shutil
import relation_util


def kr_ratio(text):
    pattern = re.compile(u"[\uac00-\ud7ff]")
    matched = pattern.findall(text)
    return len(matched) / len(text)


def jp_ratio(text):
    pattern = re.compile(u"[\u30a0-\u30ff]")
    matched = pattern.findall(text)
    return len(matched) / len(text)


def zh_ratio(text):
    pattern = re.compile(u'[\u4e00-\u9fa5]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def en_ratio(text):
    pattern = re.compile('[a-zA-Z]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def ru_ratio(text):
    pattern = re.compile(u'[\u0400-\u04ff]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def remove_html_tags(text):
    # remove comments
    comments_pattern = '<!--.*?-->'
    text = re.sub(comments_pattern, '', text)
    # remove img tag
    tags_pattern = '<imgsrc.*?/>'
    text = re.sub(tags_pattern, '', text)
    return text


def get_articles_from_db(num=None):
    database = "jdqddb"
    user = "jdqd"
    password = "jdqd"
    host = "139.9.126.19"
    port = "31001"

    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    print('connection ok')

    t1 = time.time()
    if num:
        sql = f"select * from t_article_msg limit {num}"
    else:
        sql = "select * from t_article_msg"
    cursor = conn.cursor()
    cursor.execute(sql)

    results = []
    result = cursor.fetchone()
    while result:
        print('fetching data')
        results.append(result)
        result = cursor.fetchone()
    cursor.close()
    conn.close()
    t2 = time.time()
    print(f'query used {t2 - t1} seconds')
    return results


def save_articles(results, save_dir='articles'):
    articles = [r[4] for r in results]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, a in enumerate(articles):
        if a is None:
            continue
        with open(f'{save_dir}/{i}.txt', 'w', encoding='utf-8') as f:
            f.write(a)


def load_articles(articles_dir, remove_space=True):
    articles = os.listdir(articles_dir)
    article_paths = [f'{articles_dir}/{article}' for article in articles]
    article_contents = []
    for a, ap in zip(articles, article_paths):
        with open(ap, 'r', encoding='utf-8') as f:
            content = f.read().replace('\t', '').replace('\u3000', '').replace('\n', '')
            if remove_space:
                content = content.replace(' ', '')
            article_contents.append([a, content])
    return article_contents


def get_sub_sentences(articles_dir):
    article_contents = load_articles(articles_dir)
    article_sub_sentences = {}
    for t, c in article_contents:
        sentences = relation_util.split_article_to_sentences(c)
        sub_sentences_article = []
        for s in sentences:
            min_sentences, delimiters = relation_util.split_sentence(s)
            sub_sentences = relation_util.slice_sentence(min_sentences, delimiters)
            sub_sentences_article.append([s, sub_sentences])
        article_sub_sentences[t] = sub_sentences_article
    return article_sub_sentences




def combine_files(src, dst):
    src_files = os.listdir(src)
    dst_files = os.listdir(dst)
    dst_file_nums = [int(fn.split('.')[0]) for fn in dst_files]
    max_num = max(dst_file_nums) + 1
    for i, sf in enumerate(src_files):
        shutil.move(os.path.join(src, sf), os.path.join(dst, str(max_num + i)) + '.txt')


def move_files_by_lang():
    articles = load_articles('articles_zh')
    for n, a in articles:
        kr_ratio_a = kr_ratio(a)
        jp_ratio_a = jp_ratio(a)
        en_ratio_a = en_ratio(a)
        ru_ratio_a = ru_ratio(a)
        if kr_ratio_a > 0.4:
            print('moving', n)
            shutil.move(f'articles_zh/{n}', f'articles_kr/{n}')
        if jp_ratio_a > 0:
            print('moving', n)
            shutil.move(f'articles_zh/{n}', f'articles_jp/{n}')
        if en_ratio_a > 0.4:
            print('moving', n)
            shutil.move(f'articles_zh/{n}', f'articles_en/{n}')
        if ru_ratio_a > 0.4:
            print('moving', n)
            shutil.move(f'articles_zh/{n}', f'articles_ru/{n}')


def remove_article_html_tags():
    articles = load_articles(articles_dir)
    for n, a in articles:
        text = remove_html_tags(a)
        with open(articles_dir + '/' + n, 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    articles_dir = 'articles_zh'
    move_files_by_lang()
