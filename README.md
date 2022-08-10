<h1 align="center"> Malicious URL Detection Using Deep Learning </h1> 
<h3 align="center"> Deep Learning 046211 – winter 2021 </h3>
<h5 align="center"> Technion – Israel Institute of Technology </h5>

  <p align="center">
    <a href="https://github.com/idankinderman"> Edan Kinderman </a> •
    <a href="https://github.com/levph"> Lev Panov </a> 
  </p>

<br />

<p align="center">
  <img src="https://user-images.githubusercontent.com/62880315/146927482-173cfd4d-2386-47da-92a3-e362f93f241a.gif" alt="animated" />
</p>

<br />

- [Summary](#summary)
- [Introduction](#introduction)
- [Method](#method)
- [The Models](#the-models)
- [Results](#results)
- [Files and Usage](#files-and-usage)
- [References and credits](#references-and-credits)


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="summary"> :book: Summary </h2>

In this project, we implemented and tested DL models which classify URLs as malicious or benign, based solely on lexicographical analysis of the addresses.
We used an embedding layer which maps chars from our dataset to vector representations, and we integrated it with two different models (LSTM and transformer).
From comparison of their results we concluded that the Transformer is more suitable for our task, so we created a bigger version of it which reached accuracy of 94.8% on the test set.

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="introduction"> :teacher: Introduction </h2>

Malicious URLs or malicious website is a very serious threat on cybersecurity. Malicious URLs host unsolicited content (spam, phishing, drive-by downloads, etc.) and lure unsuspecting users to become victims of scams (monetary loss, theft of private information, and malware installation), and cause losses of billions of dollars every year.

Our goal in this project is to develop DL models to precisely classify URLs as malicious or benign, and alert users if given URLS are potentially harmful. classification is based solely on lexicographical analysis of the addresses. 
Namely, it is a Binary Time Series Classification problem (many to one).

<br />


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="method"> :thought_balloon: Method </h2>

URL addresses contain multiple characteristics, which make traditional NLP methods yield non satisfying results for address text analysis.
To begin with, most URL addresses are not constructed of dictionary words, like regular sentences. 

We can try to define certain char delimiters (e.g. ‘/’,’-‘) and tokenize the addresses, thus creating word separated “sentences”.
However, tokens (words) generated from this method will not necessarily output logical or frequent words that will make a good-enough representation of the data.
Moreover, after splitting the addresses, we’ll see that many of these words are exceptionally long, extremely rare and sometimes unique in our dataset.

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169981-67ebf1d5-2e58-4902-bde6-1a4ae237ff81.png" align="center" alt="Parameters maps" width="640" height="140">
</p>

<br />

Therefore, we conclude that our project requires a method which examines the URLs char-by-char, rather than word-by-word. Hence, we used an Embedding layer to map valid chars from our dataset to vector representations.

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169802-05a3f24f-cb9b-4564-bc11-191b4cb88f2d.png" align="center" alt="Parameters maps" width="550" height="140">
</p>


<br />


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="the-models"> :computer: The Models </h2>


We implemented, trained and evaluated two models. A LSTM classifier network (FC and SoftMax layers concatenated to a LSTM network’s last hidden state), and a Transformer classifier, using same concatenation of layers at output for classification.

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151399500-8963f3f6-5cb7-4644-81f2-3bea15860467.png" align="center" alt="Parameters maps" width="810" height="270">
</p>

<br />


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="results"> :chart_with_upwards_trend: Results</h2>

Here we can see the loss, train accuracy and validation accuracy during training.
The third graphs shows the validation accuracy with increased resolution, in the epochs where the performances improved the most (the "focused epochs").

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169184-26fb59ad-15c0-4d62-a543-a785ff0ee85a.png" align="center" alt="Parameters maps" width="750" height="500">
</p>

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169532-7855aa35-e227-4edd-8caa-867da001475b.png" align="center" alt="Parameters maps" width="750" height="500">
</p>

<br />

| The model | Test accuracy | Number of parameters | Number of epochs | Train time | Inference time |
| ---------------- |:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| LSTM | 84.25% | 14,474 | 21 | 3 hours and 11 minutes | 0.01 seconds |
| Transformer | 93.39% | 11,890 | 10 | 51 minutes | 0.003 seconds |


<br />

From those measures, we conclude that the Transformer is more suitable for our classification task, so we continued to work with it.

We changed the model’s hyper-parameters so it will contain more learned parameters (27,618 in total), and we trained it for more epochs with additional URLs. Finally, model performance reached 94.8% test accuracy. Here his loss, train accuracy and validation accuracy graphs:

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151404904-e5ea89b4-12ea-4792-8679-859916a58c21.png" align="center" alt="Parameters maps" width="750" height="250">
</p>


<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="files-and-usage"> :man_technologist: Files and Usage</h2>

| File Name        | Description           |
| ---------------- |:-----------------:|
| preprocessing.py | Loads the data, preprocessing it, and creates batches |
| training_and_evaluating.py | Implementation of our training and evaluation loops |
| embedding_and_positional_encoding.py | Implementation of our embedding and positional encoding layers |
| LSTM.py | Implementation of our LSTM-classifier model |
| transformer.py | Implementation of our Transformer-classifier model |
| transformer_optuna.py | Optuna trials for our transformer |
| malicious_phish_CSV | The URL's dataset |

<br />

For running the models, you can run the files transformer.py or LSTM.py, with your own hyper-parameters.

<br />

Prerequisites
|Library         | Version |
|--------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`| `1.10.1`|
|`numpy`| `1.19.5`|
|`matplotlib`| `3.3.4`|

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* <a id="ref1">[[1]](https://www.kaggle.com/sid321axn/malicious-urls-dataset)</a> The dataset "malicious_phish.csv" was taken from kaggle.
* <a id="ref2">[[2]](https://github.com/taldatech/ee046211-deep-learning)</a> We used some of the explanation and code parts from the "deep learning - 046211" course tutorials. 
* <a id="ref3">[[3]](https://www.researchgate.net/publication/308365207_Detecting_Malicious_URLs_Using_Lexical_Analysis)</a> For more information on classification of URLs using lexical methods, see: “Detecting Malicious URLs Using Lexical Analysis”, M. Mamun, M. Rathore, A. Lashkari, N. Stakhanova, A. Ghorbani, International Conference on Network and System Security, 2016. 
