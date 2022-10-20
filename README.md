<h1>A simple use of database and crawler and machine learning</h1>

Python practice.

<h2>First part</h2>

Course similarity judge.

image similarity judge.

Lyrics similarity judge.

You can use the database operations to add, delete, update, find the course which you want to operate.

<h2>Second part</h2>

The process of course similarity judgment, firstly lower case words, secondly use the word_tokenize to separate the punctuations from the sentence and remove the punctuations and stop_words, then make the words stemmed and remove low-frequency words, finally build the TF-IDF model and compute the LSI model, then you can use the vectors which stand the courses to calculate the similarity of the courses.

<h2>Third part</h2>

The image operations are computing the histogram and gray degree, then calculate the contact ratio of different images.

<h2>Last part</h2>

The Lyrics similarity judgment uses the jieba module and crawl the data of NetEase CloudMusic, the subsequent processing is the same as the course part.

<h2>Write in the last</h2>

化学专业的菜狗啥都不会随便缝的轻喷。