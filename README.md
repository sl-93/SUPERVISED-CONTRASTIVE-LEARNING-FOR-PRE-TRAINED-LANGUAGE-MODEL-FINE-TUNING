"# SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING" 


in this code, I've implemented sentiment analysis task with sst-2 dataset.

the below results are for 100 training samples:

cross entropy loss:

![My Image](https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/result/cross%20loss.png)

cross entropy + contrastive loss:

![My Image](https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/result/cross%20%2B%20contrastive%20loss.png)


cross entropy heatmap on test dataset:

![My Image](https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/result/cross%20heatmap.png)

Accuracy on test dataset:90.13


cross entropy + contrastive loss heatmap:

![My Image](https://github.com/sl-93/SUPERVISED-CONTRASTIVE-LEARNING-FOR-PRE-TRAINED-LANGUAGE-MODEL-FINE-TUNING/blob/main/result/cross%20%2B%20contrastive%20heatmap.png)

Accuracy on test dataset:92.20



paper:
https://arxiv.org/abs/2011.01403
