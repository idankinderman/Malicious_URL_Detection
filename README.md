<h1 align="center"> Malicious URL Detection Using Deep Learning </h1> 
<h3 align="center"> Deep Learning 046211 – winter 2021 </h3>
<h5 align="center"> Technion – Israel Institute of Technology </h5>

  <p align="center">
    <a href="https://github.com/idankinderman"> Edan Kinderman </a> •
    <a href="https://github.com/levph"> Lev Panov </a> 
  </p>

<br />
<br />

<p align="center">
  <img src="https://user-images.githubusercontent.com/62880315/146927482-173cfd4d-2386-47da-92a3-e362f93f241a.gif" alt="animated" />
</p>

<br />
<br />

- [Summary](#summary)
- [Introduction](#introduction)
- [Method](#method)
- [Results](#results)
- [Files and Usage](#files-and-usage)
- [References and credits](#references-and-credits)


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="summary"> :book: Summary </h2>

?

<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)


<h2 id="introduction"> :teacher: Introduction </h2>

? 

<br />


![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="method"> :thought_balloon: Method </h2>

? 

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169981-67ebf1d5-2e58-4902-bde6-1a4ae237ff81.png" align="center" alt="Parameters maps" width="600" height="120">
</p>

<br />

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151169802-05a3f24f-cb9b-4564-bc11-191b4cb88f2d.png" align="center" alt="Parameters maps" width="550" height="140">
</p>


<br />

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="results"> :chart_with_upwards_trend: Results</h2>

? 

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

<p align="center">
<img src="https://user-images.githubusercontent.com/62880315/151363429-92249587-9cb9-4fbb-96c7-d66b9e34204d.png" align="center" alt="Parameters maps" width="750" height="250">
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

![-----------------------------------------------------](https://user-images.githubusercontent.com/62880315/143689276-058e2ec4-98ac-4367-863d-5334b959bb44.png)

<h2 id="references-and-credits"> :raised_hands: References and credits</h2>

* <a id="ref1">[[1]](https://www.kaggle.com/sid321axn/malicious-urls-dataset)</a> The dataset "malicious_phish.csv" was taken from kaggle. [↩](#dw-mri)
* <a id="ref2">[[2]](https://github.com/taldatech/ee046211-deep-learning)</a> We used some of the explanation and code parts from the "deep learning - 046211" course tutorials. 
* <a id="ref3">[[3]](https://www.researchgate.net/publication/308365207_Detecting_Malicious_URLs_Using_Lexical_Analysis)</a> For more information on classification of URLs using lexical methods, see: “Detecting Malicious URLs Using Lexical Analysis”, M. Mamun, M. Rathore, A. Lashkari, N. Stakhanova, A. Ghorbani, International Conference on Network and System Security, 2016. 
