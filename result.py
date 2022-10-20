import pandas as pd
import sqlite3 as sql
import random
import urllib.request
import urllib.parse
from jieba import lcut
from lxml import etree
import requests
import json
import re
import cv2
import numpy as np
import os

##### 初始的数据来自两组excel表格，后续再加入不需要excel表格插入数据的例子
dataset1=pd.read_excel('course.xls')
dataset2=pd.read_excel('song.xls')
dataset3 = pd.read_excel("pic1.xls")

###### 连接到一个数据库中
connect = sql.connect("work.db")
cursor=connect.cursor()

##### 这里一定要用try except，否则会因为已经存在的数据库而报错
try:
    dataset1.to_sql(
                    name='Lesson',#一参是表格名
                    con=connect,#数据库
                    chunksize=1000,#chunksize可以设置一次入库的大小
                    if_exists='fail',#‘replace'表示将表原来数据删除放入当前数据；‘append'表示追加；‘fail'则表示将抛出异常，结束操作，默认是‘fail'；
                    index=None#是否将DataFrame的index也作为表的列存储
                   )
except:
    pass
try:
    dataset2.to_sql(
                    name='songs',  # 一参是表格名
                    con=connect,  # 数据库
                    chunksize=1000,  # chunksize可以设置一次入库的大小
                    if_exists='fail',  # ‘replace'表示将表原来数据删除放入当前数据；‘append'表示追加；‘fail'则表示将抛出异常，结束操作，默认是‘fail'；
                    index=None  # 是否将DataFrame的index也作为表的列存储
                    )
except:
    pass
try:
    dataset3.to_sql(
                    name='picture',  # 一参是表格名
                    con=connect,  # 数据库
                    chunksize=1000,  # chunksize可以设置一次入库的大小
                    if_exists='fail',  # ‘replace'表示将表原来数据删除放入当前数据；‘append'表示追加；‘fail'则表示将抛出异常，结束操作，默认是‘fail'；
                    index=None  # 是否将DataFrame的index也作为表的列存储
                    )
except:
    pass

##### 以上完成了与数据库的连接与导入初始数据

##########################################################################
##### 以下是课程数据板块
def add_course(a,b,c,d,e):
    print('=='+a+'插入完成==')
    # 插入数据
    # sql = "INSERT INTO Lesson(课程名称, 学分) VALUES(\'love\', 22)"
    # cursor.execute(sql)
    # 插入数据 2
    data = (a,b,c,d,e)  # 五个列的变量
    sql = "INSERT INTO Lesson(课程名称, 学分,时间,人数,简介) VALUES(?,?,?,?,?)"
    cursor.execute(sql, data)
    # 提交事物
    connect.commit()

def del_course(name):
    print('==删除完成==')
    sql = 'delete from Lesson where 课程名称=?'
    cursor.execute(sql, (name,))
    connect.commit()

def updata_course(old, new):
    print('==更新完成==')
    sql = "UPDATE Lesson SET 课程名称 = ? WHERE 课程名称 = '%s'"%old
    cursor.execute(sql,(new,))
    connect.commit()

def find_all_course():
    list_find_all=[]
    # print('==查找全部的结果==')
    sql = "select * from Lesson"
    values = cursor.execute(sql)
    for i in values:
        # print(i)
        list_find_all.append(i)
    return list_find_all

def find_one_course(name):
    print('==查找一个的结果==')
    sql = "select * from Lesson where 课程名称=?"
    values = cursor.execute(sql, (name,))
    for i in values:
        print('====')
        print('课程名称:', i[0])
        print('学分:', i[1])
        print('时间:', i[2])
        print('人数:', i[3])
        print('简介:', i[4])

    # 提交事物
    connect.commit()

# 这里导入大量的课程数据，数据来源为助教pdf中提供的链接
def mass_data():
    # 打开方式编码为'utf-8'否则可能会出现乱码
    fp=open('coursera_corpus.txt',encoding='utf-8')

    # 因为数据中一个课程名称及其简介同其他课程是以换行符区分开，因此以行读取文件
    courses = [line.strip() for line in fp]

    # 处理读取的课程名称和简介
    courses_name = [course.split('\t')[0] for course in courses]
    courses_in = [course.split('\t')[1:] for course in courses]
    courses_intr=[" ".join(course) for course in courses_in]
    print(courses_name[0:10])
    print(courses_intr[0:10])
    print(len(courses_name))
    print(len(courses_intr))

    # 这里集体导入数据库，并且为学分，时间和人数生成随机数填充，如果不生成随机数填充则默认以NULL填充，问题不大
    for i in range(len(courses_name)):
        add_course(courses_name[i],random.randint(1,6),random.randint(10,40),random.randint(30,300),courses_intr[i])

##### 以上是课程数据板块
#################################################

#################################################
##### 以下是图片数据板块
def add_pic(name,img):
    # 注意使用Binary()函数来指定存储的是二进制
    # cursor.execute("insert into img set imgs='%s'" % mysql.Binary(img))
    sql1 = "INSERT INTO picture(名称,图片) VALUES(?,?)"
    cursor.execute(sql1, (name,img))
    print('存入成功')
    # 如果数据库没有设置自动提交，这里要提交一下
    connect.commit()

def del_pic(name):
    print('==删除完成==')
    sql = 'delete from picture where 名称=?'
    cursor.execute(sql, (name,))
    connect.commit()
    os.remove('pic_sample/'+name)

##### 以上是图片数据板块
####################################################

####################################################
##### 以下是歌词数据板块（中文）
def get_songinfo(song_id):
    # song_name=input('请输入歌名:（可以用空格隔开歌名与歌手）')
    # 由于网抑云的网页大部分数据都是用的Ajax请求，所以基本不能直接通过进入网页显示的网址获取到想要的信息
    # 而且网抑云首页的url带井号，一般来说这就是个假的url，程序并不能通过这个url得到需要的响应数据
    # 在多次解析网页的post的请求后，尝试了8次的响应数据中url代入回发起请求过程，最终才能得到歌曲的数据和歌词——plan A

    # 啊但是之后通过万能的群友得到了一个老网站，可以直接根据id得到歌词数据，于是plan A被舍弃了

    # 这里进行U-A伪装然后发起请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
    url = 'http://music.163.com/api/song/lyric?'+'id='+str(song_id)+'&lv=1&kv=1&tv=-1'
    response = requests.get(url=url, headers=headers)
    # 将json格式的响应数据读取出来
    page_text=response.text
    j=json.loads(page_text)

    # 通过一般的url请求得到歌词以外的歌名等其他信息，并通过xpath或bs4解析
    url1='https://music.163.com/song?id='+str(song_id)
    j1=requests.get(url=url1,headers=headers).text
    tree=etree.HTML(j1)

    song_name=tree.xpath('/html/head/title//text()')[0].split()[0]
    print(song_name,"添加完毕")

    # 这个url用于获取歌曲文件
    url2="http://music.163.com/song/media/outer/url?id=" + str(song_id)
    r=requests.get(url=url2,headers=headers)
    # 获取歌曲、图片等二进制数据使用content而不是text
    song_content=r.content

    # 若不存在文件则创建文件存储
    if not os.path.exists('./'+'music'):
        os.mkdir('./'+'music')
    fname='music/'+song_name+'.mp3'

    # 使用'wb'来写入二进制数据
    with open(fname,'wb') as file:
        file.write(song_content)


    # 将j读取到的json数据中需要的数据取出，并通过正则表达式去掉不需要的词缀，最后去掉前后空白
    lrc=j['lrc']['lyric']
    pat=re.compile(r'\[.*\]')
    lrc=re.sub(pat,"",lrc)
    lrc=lrc.strip()

    # 创建文件存储，注意编码为UTF-8否则容易乱码
    if not os.path.exists('./' + 'lyric'):
        os.mkdir('./' + 'lyric')
    filename='lyric/'+song_name+'.txt'
    with open(filename,'w',encoding='utf-8') as fp:
        fp.write(song_name+'\n')
        fp.write(lrc)
    print("ALL OK")

    # 同时将数据储存到数据库中的另一个表格中
    data=(song_name,lrc)
    sql = "INSERT INTO songs(歌名,歌词) VALUES(?,?)"
    cursor.execute(sql, data)
    connect.commit()

def delete_songinfo(name):
    # 这里就不需要id了，对于储存到数据库中的歌曲只用名字就可以删除了
    print('==删除完成==')
    sql = 'delete from songs where 歌名=?'
    cursor.execute(sql, (name,))
    connect.commit()
    os.remove('lyric/'+name+'.txt')
    os.remove('music/'+name+'.mp3')

def find_one_song(name):
    # 查找操作
    print('==查找一个的结果==')
    sql = "select * from songs where 歌名=?"
    values = cursor.execute(sql, (name,))
    for i in values:
        print('====')
        print('歌名:', i[0])
        print('歌词:', i[1])
    # 提交事物
    connect.commit()

def find_all_song():
    # 查找全部
    list_find_all = []
    print('==查找全部的结果==')
    sql = "select * from songs"
    values = cursor.execute(sql)
    for i in values:
        list_find_all.append(i)
    print(len(list_find_all))
    return list_find_all

##### 以上是歌词数据板块（中文）
#######################################################

# 读取所有课程数据，以便后续使用
all_course=find_all_course()
db_courses = []
db_courses_name = []
for i in all_course:
    str1 = i[0] + ' ' + i[-1]
    db_courses.append(str1)
    db_courses_name.append(i[0])
# 数据读取完成
# print(db_courses_name)
# print(db_courses)

###################################################
##### 以下是英文相似识别板块
def nl(index1):
    # 小写处理
    texts_lower = [[word for word in document.lower().split()] for document in db_courses]
    print(texts_lower[0])

    # 导入函数分开混在单词中的标点符号
    from nltk.tokenize import word_tokenize
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in db_courses]
    print(texts_tokenized[0])

    # 导入英文的部分停词
    from nltk.corpus import stopwords
    english_stopwords = stopwords.words('english')
    print(english_stopwords)

    # 去掉词组中的停词
    texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in
                                texts_tokenized]
    print(texts_filtered_stopwords[0])

    # 去掉标点
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in
                      texts_filtered_stopwords]
    print(texts_filtered[0])

    # 导入干词处理的函数
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()
    print(st.stem('founded'))

    # 干词处理
    texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
    print(texts_stemmed[0])

    # 低频词
    all_stems = sum(texts_stemmed, [])
    stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    print(stems_once)

    # 去掉低频词
    texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
    print(texts[0])

    # 导入相关的函数进行后续处理
    from gensim import corpora, models, similarities
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 建立词组字典
    dictionary = corpora.Dictionary(texts)

    # 基于词典，建立稀疏向量集，也就是语料库，后续的语言处理都是基于该语料库进行的
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 建立TF-IDF模型，TF-IDF是词频-逆文件频率
    # 一个词的重要性会随着这个词在一个文件中出现的次数增加而增加；也会随着这个词在整个语料库的出现次数增加而减少
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 计算LSI模型，将这些高维数据降维，topic模型多试了几次还是选了30左右
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=30)
    index = similarities.MatrixSimilarity(lsi[corpus])

    # 打印一下我们要预测的那个课程名称
    print(db_courses_name[index1])

    # 开始处理
    ml_course = texts[index1]
    ml_bow = dictionary.doc2bow(ml_course)
    ml_lsi = lsi[ml_bow]
    print(ml_lsi)

    # 计算相似度并按相似程度排序
    sims = index[ml_lsi]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims[:10])
    sim_course = []
    for i in sort_sims[:10]:
        sim_course.append(db_courses_name[i[0]])
    print('最相似的几门课程为:', end='')
    print(sim_course)

##### 以上是英文相似识别板块
############################################################

# 读取所有的图片数据，以便后续处理
all_pic = []
sql_pic = "select * from picture"
values = cursor.execute(sql_pic)
for i in values:
    # 图片用二进制存储，这里一定要进行处理
    j=cv2.imdecode(np.asarray(bytearray(i[1]),dtype='uint8'), cv2.IMREAD_COLOR)
    all_pic.append((i[0],j))

#############################################################
##### 以下是图片相似处理识别板块
def classify_gray_hist(image1, image2, size=(256, 256)):
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    # bins 取为16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def pic_similar(index1):
    print('将要作比较的图片为'+ all_pic[index1][0])
    degree_now=0
    for i in range(0,len(all_pic)):
        if i!=index1:
            degree=classify_gray_hist(all_pic[index1][1],all_pic[i][1])
            if degree>degree_now:
                degree_now=degree
                index2=i
    print('与'+all_pic[index1][0]+'最相似的图片为'+all_pic[index2][0])
    print('相似度为'+str(degree_now))

##### 以上是图片相似处理识别板块
##########################################################

##########################################################
##### 以下是英汉互译板块
def translate(content):
    # 这里添上需要爬取的对应的url，此处使用有道翻译因此添上有道翻译的网址
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

    # 这里进行U-A伪装，否则容易被墙
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'

    # 这里是需要传入的数据，其实大部分数据都不用改动，在网页的开发者工具页面下能够得到这些
    # 在进行了几个示例翻译后，发现了这些数据中发生变化的之后'i'中的值，因此只需要把待翻译内容传入此处即可
    data = {}
    data['i'] = content
    data['from'] = 'en'
    data['to'] = 'zh-CHS'
    data['smartresult'] = 'dict'
    data['client'] = 'fanyideskweb'
    data['salt'] = '16162486242827'
    data['sign'] = '7929e50a6efbf84117a46eb6e5ee71b9'
    data['lts'] = '1616248624282'
    data['bv'] = '8d869977ed9730c759a83d50a1f65ed0'
    data['doctype'] = 'json'
    data['version'] = '2.1'
    data['keyfrom'] = 'fanyi.web'
    data['action'] = 'FY_BY_CLICKBUTTION'

    # 这里进行'UTF-8'的编码是为了防止出现乱码
    data = urllib.parse.urlencode(data).encode('utf-8')

    # 发起请求，获取响应数据
    req = urllib.request.Request(url, data, head)
    response = urllib.request.urlopen(req)
    # 读取响应数据仍然使用'UTF-8'解码
    html = response.read().decode('utf-8')

    # 得到的响应数据应是json格式，但爬取的只是有json的格式的字符串，因此进行json.loads()读取json数据
    target = json.loads(html)
    # 找到数据中对应的翻译后的语句，作为返回值返回
    return str(target['translateResult'][0][0]['tgt'])

##### 以上是英汉互译板块
##########################################################

# 读取所有的歌词数据，以便后续使用
all_song=[]
sql_song = "select * from songs"
values = cursor.execute(sql_song)
for i in values:
    all_song.append(i)

##########################################################
##### 以下是中文歌词相似推荐板块
def jb_sim(index1):
    from gensim.corpora import Dictionary
    from gensim import models, similarities
    # 将歌名与歌词分开
    db_lyric = []
    db_name = []
    for i in all_song:
        db_name.append(i[0])
        db_lyric.append(i[0] + " " + i[-1])
    print(db_name)
    print(db_lyric)

    # 得到的歌词有换行符，去掉换行符
    new_lyric=[]
    for i in db_lyric:
        str=" ".join(i.split('\n'))
        new_lyric.append(str)
    print(new_lyric[index1])

    # 生成分词列表
    texts = [lcut(text) for text in new_lyric]
    print(texts[index1])

    # 去掉歌词中的空格
    for i in texts:
        while ' ' in i:
            i.remove(' ')

    # 去掉中文歌词中一些奇奇怪怪的符号，得到最终的文本集
    english_punctuations = ['/','，', '。', '：', '；', '？', '（', '）', '【', '】', '&', '！', '*', '@', '#', '$', '%', '~',':','_','《','》','(',')']
    texts = [[word for word in document if not word in english_punctuations] for document in
                      texts]
    print(texts[index1])

    # 基于文本集建立词典，并获得词典特征数
    dictionary = Dictionary(texts)
    num_features = len(dictionary.token2id)
    print(dictionary)
    print(num_features)

    # 基于词典建立新的语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('语料库：', corpus[index1])

    # 建立TF-IDF模型，TF-IDF是词频-逆文件频率
    # 一个词的重要性会随着这个词在一个文件中出现的次数增加而增加；也会随着这个词在整个语料库的出现次数增加而减少
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # 计算LSI模型，将这些高维数据降维
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=15)
    index = similarities.MatrixSimilarity(lsi[corpus])

    # 打印一下我们要预测的歌曲名称
    print(db_name[index1])

    # 开始处理
    ml_lyric = texts[index1]
    ml_bow = dictionary.doc2bow(ml_lyric)
    ml_lsi = lsi[ml_bow]
    print(ml_lsi)

    # 计算相似度并按相似度排序
    sims = index[ml_lsi]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims[0:10])

    sim_song = []
    for i in sort_sims[:10]:
        sim_song.append(db_name[i[0]])
    print('最相似的几首推荐歌曲为:', end='')
    print(sim_song)

##### 以上是中文歌词相似推荐板块
##########################################################



##########################################################
##########################################################
##########################################################
# 以下是你可以手动操作的地方


# add_course('HAHA',2,3,2,'haoye')
# del_course('HAHA')
# updata_course('HAHA','haha')
# find_one_course('Machine Learning')
# c=find_all_course()
# print(c)
# mass_data()
# nl(215)


# fil='pic_sample/'
# pic_name='70.png'
# filename=fil+pic_name
# fp1=open(filename,'rb')
# fp=fp1.read()
# img = cv2.imread(filename)
#
# add_pic(pic_name,fp)
# del_pic('70.png')
# pic_similar(10)


# trans_res=translate("我觉得这件事更重要")
# print(trans_res)


# get_songinfo(504914212)
# delete_songinfo('不离')
# find_one_song('不老梦')
# b=find_all_song()
# print(b)
# jb_sim(75)

################################################
##### 关闭数据库
cursor.close()
connect.close()

