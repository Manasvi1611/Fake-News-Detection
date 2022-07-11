[Copy_of_fakeNewsDetection (1).ipynb - Colaboratory.pdf](https://github.com/Manasvi1611/Fake-News-Detection/files/9084817/Copy_of_fakeNewsDetection.1.ipynb.-.Colaboratory.pdf)
 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns from wordcloud 
import WordCloud 
import nltk
import re from wordcloud import WordCloud
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.text 
import Tokenizer from tensorflow.keras.preprocessing.sequence
import pad_sequences from tensorflow.keras.utils 
import to_categorical from tensorflow.keras import regularizers from tensorflow.keras.models 
import Sequential from tensorflow.keras.layers 
import Dense,Embedding,LSTM,Conv1D,MaxPool1D 
from sklearn.metrics import classification_report,accuracy_score
Explore fake news dataset
#explore fake dataset fake = pd.read_csv("/content/Fake_data.csv")
 
fake.head()
title	text subject	date
 
fake.columns 
Index(['title', 'text', 'subject', 'date'], dtype='object')
#types of subjects in fake data
fake["subject"].value_counts() 
plt.figure(figsize =(10,6)) sns.countplot(x = "subject" ,data = fake) 
<matplotlib.axes._subplots.AxesSubplot at 0x7efdc7519690>
 
WORLDCLOUD
 #convert text column into list  ls = fake["text"].tolist()  #join the list into a string  text = " ".join(ls)  wordcloud = WordCloud(width = 1920,height = 1080).generate(text)  fig = plt.figure(figsize = (10,20))  plt.imshow(wordcloud)  plt.axis("off")  plt.tight_layout(pad = 0)  plt.show() 
0	Republicans flip t...	head of a conservat...	politicsNews	31, 2017
1	U.S. military to accept transgender recruits o...	WASHINGTON (Reuters) Transgender people will...	politicsNews	December 29, 2017
2	Senior U.S. Republican senator: 'Let Mr. Muell...	WASHINGTON (Reuters) - The special counsel inv...	politicsNews	December 31, 2017
3	FBI Russia probe helped by	WASHINGTON (Reuters) - Trump	liti	N	December
 
#columns in true data set true.columns 
Index(['title', 'text', 'subject', 'date'], dtype='object')
#types of subjects in true data true["subject"].value_counts() 
politicsNews    11272 worldnews       10145 
Name: subject, dtype: int64
plt.figure(figsize =(10,6)) sns.countplot(x = "subject" ,data = true) 
 
plt.imshow(wordcloud) plt.axis("off") plt.tight_layout(pad = 0) plt.show() 
 
Difference in Text Real news seems to have source of publications which is not present in fake dataset Looking at the data:
 most of the text contains reuters information such as "WASHINGTON (reuters)"
Some text are tweets from twitter.
Few texts do not contain any publication information
Cleaning Of Data Removing reuters or twitter tweet information from the text .
 text can b.e splitted only once at "-" which is always present after mentioning source of publication, this gives us publication part and the text part
If we do not get the text part , this means publication details was not given for that record.
The Twitter tweets always have some source, a long text of max 259 characters
true.sample(5) 
title	text	subject	date
 
3301	Former FBI chief's ire over Trump laid bare in...	WASHINGTON (Reuters) Throughout the drama of...	politicsNews	June 8, 2017
1828	Moderate Republican U.S. congressman Dent will...	WASHINGTON (Reuters) - U.S. Representative Cha...	politicsNews	September 8, 2017
11214	House committee seeks testimony from 'Pharma B...	WASHINGTON (Reuters) - A U.S. congressional pa...	politicsNews	January
15, 2016
5512	Japan's love of tiny cars sore	TOKYO/DETROIT (Reuters) -	liti	N	February
# creating a list of indexes that do not have publications unknown_publishers = [] for index, row in enumerate(true.text.values):   try: 
    record = row.split("-", maxsplit = 1)     record[1]     assert(len(record[0])<120)   except:     unknown_publishers.append(index) 
len(unknown_publishers) 
222
#list the text of unknown publishers true.iloc[unknown_publishers].text 
7	The following statements were posted to the ve... 
8	The following statements were posted to the ve... 
12	The following statements were posted to the ve... 
13	The following statements were posted to the ve... 
14	(In Dec. 25 story, in second paragraph, corre... 
                               ...                         
20135     (Story corrects to million from billion in pa... 
20500     (This Sept 8 story corrects headline, clarifi... 
20667     (Story refiles to add dropped word  not , in ... 
21246     (Story corrects third paragraph to show Mosul... 
21339     (Story corrects to fix spelling in paragraph ... Name: text, Length: 222, dtype: object
#list of known publishers publishers = [] tmp_text =[] 
for index , row in enumerate (true.text.values):   if index in unknown_publishers:     tmp_text.append(row)     publishers.append("unknown") 
       else: 
      record = row.split("-",maxsplit = 1)       publishers.append(record[0].strip())       tmp_text.append(record[1].strip()) 
true["publishers"] = publishers true["text"] = tmp_text 
true.head() 
title	text	subject	date	publishers
 
0	As U.S. budget fight looms, Republicans flip t...	The head of a conservative Republican faction ...	politicsNews	December 31, 2017	WASHINGTON (Reuters)
1	U.S. military to accept transgender recruits o...	Transgender people will be allowed for the fir...	politicsNews	December 29, 2017	WASHINGTON (Reuters)
	Senior U.S. Republican	The special counsel
i	i	i	f li k		December	
true.shape 
(21417, 5)
#to check whether fake news dataset has any empty text field  
empty_fake_index = [index for index,text in enumerate(fake.text.tolist()) if str(text).str fake.iloc[empty_fake_index] 
title text subject	date
 
May 10, 10923 TAKE OUR POLL: Who Do You Think President Trum... politics
2017
Apr 26,
11041	Joe Scarborough BERATES Mika Brzezinski Over “...	politics
2017 11190	WATCH TUCKER CARLSON Scorch Sanctuary City May...	politics	Apr 6, 2017
11225	MAYOR OF SANCTUARY CITY: Trump Trying To Make ...	politics	Apr 2, 2017
11236	SHOCKER: Public School Turns Computer Lab Into...	politics	Apr 1, 2017
...	...	...	...	...
BALTIMORE BURNS: MARYLAND GOVERNOR BRINGS left- Apr 27, #the text21816 of these rows seems to be present in the title IN N... news 2015
#removing the rows with empty text field 
true["text"21826] = true[FULL VIDEO: THE BLOCKBUSTER INVESTIGA"title"] + " " + true["text"] 	TION	left-	Apr 25, fake["text"] = fake["title"] + " " + fake["text"] 	INTO...	news	2015
left-	Apr 25, 21827	(VIDEO) HILLARY CLINTON: RELIGIOUS BELIEFS MUS...
news	2015
#preprocessing  
#converting all the(VIDEO)ICE text to PROTECTINGlower	OBAMA: WON’T RELEASE	left-	Apr 14
true['text'] = true['text'].apply(lambda x:str(x).lower()) fake['text'] = fake['text'].apply(lambda x:str(x).lower()) 
PREPROCESSING TEXT
#assigning labels to the news true['class'] = 1 fake['class'] = 0 
true.columns 
Index(['title', 'text', 'subject', 'date', 'publishers', 'class'], dtype='object')
#getting the required columns 
Ttrue = true [['text' , 'class']] 
Ffake = fake [['text' , 'class']] 
#append the true and the fake and true data together data = Ttrue.append(Ffake,ignore_index = True) 
!pip install spacy==2.2.3 
!python -m spacy download en_core_web_sm 
!pip install beautifulsoup4==4.9.1 
!pip install textblob==0.15.3 
!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheel Collecting spacy==2.2.3 
  Downloading spacy-2.2.3-cp37-cp37m-manylinux1_x86_64.whl (10.4 MB) 
     |████████████████████████████████| 10.4 MB 4.5 MB/s  
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist
Collecting thinc<7.4.0,>=7.3.0 
  Downloading thinc-7.3.1-cp37-cp37m-manylinux1_x86_64.whl (2.2 MB) 
     |████████████████████████████████| 2.2 MB 38.3 MB/s  
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/d
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.7
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dis
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/di
Requirement already satisfied: srsly<1.1.0,>=0.1.0 in /usr/local/lib/python3.7/dis
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3
Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-pack
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-package
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.7/dist
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7 Requirement already satisfied: importlib-metadata>=0.20 in /usr/local/lib/python3.
Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/loc
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packa
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist
Requirement already satisfied: tqdm<5.0.0,>=4.10.0 in /usr/local/lib/python3.7/dis
Installing collected packages: thinc, spacy 
  Attempting uninstall: thinc 
    Found existing installation: thinc 7.4.0 
    Uninstalling thinc-7.4.0:       Successfully uninstalled thinc-7.4.0 
  Attempting uninstall: spacy 
    Found existing installation: spacy 2.2.4 
    Uninstalling spacy-2.2.4: 
      Successfully uninstalled spacy-2.2.4 
Successfully installed spacy-2.2.3 thinc-7.3.1 
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheel
Collecting en_core_web_sm==2.2.5 
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_
     |████████████████████████████████| 12.0 MB 5.2 MB/s  
Requirement already satisfied: spacy>=2.2.2 in /usr/local/lib/python3.7/dist-packa
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/d
Requirement already satisfied: srsly<1.1.0,>=0.1.0 in /usr/local/lib/python3.7/dis
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.7/dist
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.7
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dis
Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-pack
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.7/dist
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.7/di
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-package
Requirement already satisfied: thinc<7.4.0,>=7.3.0 in /usr/local/lib/python3.7/dis Requirement already satisfied: importlib-metadata>=0.20 in /usr/local/lib/python3.
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-
#removing special characters  import preprocess_kgptalkie as ps data["text"] = data["text"].apply(lambda x : ps.remove_special_chars(x)) 
converting text data into numerical data
Word2vec is one of the most popular technique to learn word embeddings using shallow neural network. It was developed by Thomas Mikolov in 2013 at Google. Word embedding is one of the most popular representation of the document vocabulary.It is capable of capturing the context of the word in a document , sematics and syntactic similarity ,relation with other words etc. 
 
# word to vec using gensim  library import gensim 
y = data["class"].values #x is list of words x = [d.split() for d in data["text"].tolist()]
# word will be converted into sequence of 100 vectors DIM = 100 
w2v_model = gensim.models.Word2Vec(sentences = x,size = DIM ,window = 10, min_count = 1) 
len(w2v_model.wv.vocab) 
231911
w2v_model.wv["love"] 
array([ 1.2013565e-01, -5.6633401e+00, -1.7205703e+00, -2.2613537e+00,        -1.7574425e+00, -1.1907017e+00, -2.9603314e+00, -2.9502466e+00, 
       -5.9338445e-03, -6.2388766e-01,  7.2647445e-02, -3.0730557e-01, 
        1.0497167e+00,  1.7028066e+00,  9.2080766e-01, -1.3230104e+00,        -1.3527319e+00,  5.1679009e-01, -3.9076610e+00, -1.6151338e+00, 
        5.9745927e+00,  2.3901677e+00,  9.4603229e-01, -2.4719667e-02,        -1.5349996e+00,  1.0509170e-01,  1.1369998e+00, -3.1390619e-01, 
        5.3761464e-01, -8.5476118e-01, -1.8041219e-01,  1.9248329e+00,         1.1323432e+00, -3.3786154e-01,  1.3780036e+00, -3.4079230e-01,         2.9695737e-01, -1.1189826e+00, -1.6292790e-01,  1.8354844e-01,        -7.9905939e+00, -5.9293962e-01, -1.6561116e+00, -2.2430110e+00, 
       -1.8228278e-01,  1.2312325e+00, -1.1956246e+00, -1.6377302e+00, 
       -2.2719910e+00, -2.6411080e+00, -2.7145877e+00, -1.1955173e+00, 
       -2.6841050e-01,  1.1252708e+00,  3.4741735e+00,  8.3677113e-01, 
        4.1381936e+00,  4.1649752e+00,  2.0603330e+00, -1.4148881e+00, 
       -3.2231870e-01,  3.9128239e+00,  1.2981997e+00,  1.1007769e+00, 
        2.3374560e+00,  1.3240238e+00,  2.0035989e+00,  1.9537610e+00,        -1.3156407e+00, -2.1354871e+00,  5.0499183e-01, -8.5143739e-01, 
        5.7474488e-01,  1.4701848e+00, -2.4309187e+00,  8.2036841e-01,        -2.8157666e-01,  4.0025535e+00, -4.6934095e-01,  1.4898502e+00, 
       -2.6167941e+00,  2.3085673e+00, -1.0946405e+00, -1.1268820e+00, 
        1.1394301e-01,  4.4136459e-01,  1.8533053e+00, -3.3242919e+00,        -1.0166962e+00, -2.9392022e-01, -4.5921788e+00, -1.7507882e+00, 
       -2.7463322e+00, -1.1533693e+00,  1.4961511e+00, -3.3871419e+00,         2.1474388e+00, -7.5975615e-01, -1.1989495e+00,  4.3086038e+00],       dtype=float32)
#exploring the vectors w2v_model.wv["india"] w2v_model.wv.most_similar("india") 
[('pakistan', 0.7431882619857788), 
 ('malaysia', 0.7045928835868835), 
 ('indias', 0.6483880281448364), 
 ('china', 0.6466995477676392), 
 ('norway', 0.635319709777832), 
 ('indian', 0.632917046546936), 
 ('beijings', 0.6272697448730469), 
 ('australia', 0.5987348556518555), 
 ('maritime', 0.5867645144462585),  ('senegal', 0.5858911871910095)]
w2v_model.wv.most_similar("china") 
[('beijing', 0.8490604758262634),
 ('taiwan', 0.8045223355293274), 
 ('chinas', 0.7632561922073364), 
 ('chinese', 0.701541543006897), 
 ('pyongyang', 0.6801574230194092), 
 ('waterway', 0.6526803970336914), 
 ('beijings', 0.6491953134536743), 
 ('india', 0.6466995477676392), 
 ('japan', 0.6399459838867188),  ('xi', 0.6283771991729736)]
w2v_model.wv.most_similar("modi") 
[('narendra', 0.7596169114112854), 
 ('modis', 0.6608402729034424), 
 ('india', 0.5744475722312927), 
 ('najib', 0.5427131652832031), 
 ('abe', 0.5376186370849609), 
 ('premier', 0.5358973741531372),
 ('usindia', 0.5321942567825317),
 ('movetrump', 0.5306423306465149), 
 ('indias', 0.5250353217124939), 
 ('tokyo', 0.5200173854827881)]
w2v_model.wv.most_similar("trump") 
[('trumps', 0.7310428619384766), 
 ('trumpthe', 0.5766953229904175), 
 ('him', 0.5370162129402161), 
 ('trumptrump', 0.5348323583602905), 
 ('smilefeatured', 0.5293625593185425), 
 ('he', 0.5291314125061035), 
 ('rumsfeld', 0.52315753698349), 
 ('presidentelect', 0.5201758146286011), 
 ('cruz', 0.5166608095169067), 
 ('trumpfeatured', 0.5157501697540283)]
To train the dataset we will will feed the vectors as initial weight to our training model
tokenizer = Tokenizer() tokenizer.fit_on_texts(x) 
#vectors to sequence for the dataset x = tokenizer.texts_to_sequences(x) #x 
#tokenizer.word_index 
{'the': 1, 
 'to': 2, 
 'of': 3, 
 'a': 4, 
 'and': 5, 
 'in': 6, 
 'that': 7, 
 'on': 8, 
 'for': 9,  's': 10, 
 'is': 11, 
 'he': 12, 
 'said': 13, 
 'trump': 14, 
 'it': 15, 
 'with': 16, 
 'was': 17,  'as': 18, 
 'his': 19, 
 'by': 20, 
 'has': 21, 
 'be': 22, 
 'have': 23,  'not': 24, 
 'from': 25, 
 'this': 26, 
 'at': 27, 
 'are': 28, 
 'who': 29, 
 'us': 30, 
 'an': 31, 
 'they': 32, 
 'i': 33, 
 'but': 34, 
 'we': 35, 
 'would': 36, 
 'president': 37, 
 'about': 38, 
 'will': 39, 
 'their': 40, 
 'had': 41, 
 'you': 42, 
 't': 43, 
 'been': 44, 
 'were': 45, 
 'people': 46, 
 'more': 47, 
 'or': 48, 
 'after': 49, 
 'which': 50, 
 'she': 51, 
 'her': 52, 
 'one': 53,  'if': 54, 
 'its': 55, 
 'out': 56, 
 'all': 57, ' h t' 8
#histogram to show total numbers of words in the news plt.hist([len(i) for i in x],bins = 700) plt.show() 
 
#truncate news with more than 1000 words nos = np.array([len(i) for i in x]) nos[nos>1000] len(nos[nos>1000]) 
1584
maxlen = 1000 
x = pad_sequences(x,maxlen = maxlen) 
len(x[101]) 
1000
feeding the vectors as initial weight to the training model
vocab_size = len(tokenizer.word_index)+1 vocab = tokenizer.word_index 
#getting the weight matrix def get_weight_matrix(model): 
  weight_matrix = np.zeros((vocab_size,DIM))   for word, i in vocab.items(): 
    weight_matrix[i] = model.wv[word] 
     return weight_matrix 
embedding_vectors = get_weight_matrix(w2v_model) 
#shape of embediding vector embedding_vectors.shape 
(231912, 100)
#model architecture model = Sequential() 
model.add(Embedding(vocab_size,output_dim = DIM,weights = [embedding_vectors],input_length model.add(LSTM(units = 128)) model.add(Dense(1,activation = "sigmoid")) model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["acc"]) 
model.summary() 
Model: "sequential" 
_________________________________________________________________ 
 Layer (type)                Output Shape              Param #    =================================================================  embedding (Embedding)       (None, 1000, 100)         23191200                                                                      lstm (LSTM)                 (None, 128)               117248                                                                        dense (Dense)               (None, 1)                 129                                                                          
================================================================= 
Total params: 23,308,577 
Trainable params: 117,377 
Non-trainable params: 23,191,200 
_________________________________________________________________ 
#splitting the dataset into training and testing data x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) model.fit(x_train, y_train,validation_split = 0.3,epochs = 6) 
Epoch 1/6 
688/688 [==============================] - 863s 1s/step - loss: 0.0657 - acc: 0.9778 Epoch 2/6 
688/688 [==============================] - 863s 1s/step - loss: 0.0394 - acc: 0.9873 Epoch 3/6 
688/688 [==============================] - 860s 1s/step - loss: 0.0228 - acc: 0.9929 Epoch 4/6 
688/688 [==============================] - 860s 1s/step - loss: 0.0183 - acc: 0.9943 
Epoch 5/6 
688/688 [==============================] - 856s 1s/step - loss: 0.0108 - acc: 0.9970 Epoch 6/6 
688/688 [==============================] - 864s 1s/step - loss: 0.0233 - acc: 0.9927 
<keras.callbacks.History at 0x7efd5fed7b50>
 
#validating training data y_pred_Train= (model.predict(x_train) >= 0.5).astype(int) 
#validatimg test data y_pred_Test = (model.predict(x_test) >= 0.5).astype(int)
#classification report of training data print(classification_report(y_train,y_pred_Train)) 
#classification report of testing data print(classification_report(y_test,y_pred_Test)) 
#converting the random news into sequences rand= ["this is a news"] 
rand = tokenizer.texts_to_sequences(rand) rand = pad_sequences(rand,maxlen = maxlen) 
#model being used to predict the result result = (model.predict(rand) >= 0.5).astype(int) result 
array([[0]])
if result[[0]] == 0: 
  print("false") else:   print("True") false 
rand = ["India today reported 3,714 new coronavirus infections, taking the tally of COVIDrand = tokenizer.texts_to_sequences(rand) rand = pad_sequences(rand,maxlen = maxlen) result = (model.predict(rand) >= 0.5).astype(int) result if result[[0]] == 0: 
  print("false") else:   print("True") 
