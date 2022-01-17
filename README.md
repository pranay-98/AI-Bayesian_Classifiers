# AI-Bayesian_Classifiers

## Approach to : Truth be Told

<h3>Problem statement</h3>

<p>Many practical problems involve classifying textual objects — documents, emails, sentences, tweets, etc. —
into two specific categories — spam vs nonspam, important vs unimportant, acceptable vs inappropriate,
etc. Naive Bayes classifiers are often used for such problems. They often use a bag-of-words model, which
means that each object is represented as just an unordered “bag” of words, with no information about the
grammatical structure or order of words in the document. Suppose there are classes A and B. For a given
textual object D consisting of words w1, w2, ..., wn, a Bayesian classifier evaluates decides that D belongs to
A by computing the “odds” and comparing to a threshold</p>

<p>where P (A|w1, ...wn) is the posterior probability that D is in class A. Using the Naive Bayes assumption,
the odds ratio can be factored into P (A), P (B), and terms of the form P (wi|A) and P (wi|B). These are
the parameters of the Naive Bayes model.
As a specific use case for this assignment, we’ve given you a dataset of user-generated reviews. User-generated
reviews are transforming competition in the hospitality industry, because they are valuable for both the guest
and the hotel owner. For the potential guest, it’s a valuable resource during the search for an overnight stay.
For the hotelier, it’s a way to increase visibility and improve customer contact. So it really affects both the
business and guest if people fake the reviews and try to either defame a good hotel or promote a bad one.
Your task is to classify reviews into faked or legitimate, for 20 hotels in Chicago.</p>


<h3>Approach<h3>
<p>Initially, prepocessing the data by cleaning the train_data by removing stop words(most frequently used english word) which may result into wrong classification. Converting everything into lowercase letters by removing alpha-numeric characters in the train_data which results in improving accuracy of classification.</p><br>
<p>After cleaning the data, maintaining the occurrences of each words with label in a dictionary to look up when needed.</p><br>
<p>Applying Baye's rule to calculate posterior probability P(label|review_words) <i>i.e P(label|review_words)= P(review_words|label) *P(label). </i></p><br>
<p>Here, labels are Truth and deceptive. We ignore denominator P(review_words) in Bayes rule, because it is the same for both labels.</p>
<br>
<p>Based on the independence assumption in Baye's law, P(review_words|label) can be written as the product: P(w1|label)*.... * P(wn|label), for all words in the review. P(w1|label) is the frequency of word associated with the label, divided by all words associated with the label.
</p><br>
<p>In our first attempt, we got an classification accuracy of around 52%.</p>
<p>To improve accuracy and to handle zero word occurrencies which result in zero probability, used <i>Laplace smoothing</i> technique. Which will push the likelihood towards a values of 0.5 when using higher alpha values(1 in my case) and denominator is added with k*aplha where k is 2 i.e., the probability of a word equal to 0.5 for both the labels.</p><br>
<p>In our second attempt after introducing Laplace smmothing, we got an classification accuracy of around 78%.</p><br>
<p>We found that multiplying very small numbers will lead to even smaller numbers. To avoid those tried using log probabilities, which helped us in increasing accuracy to 85.75%.</p>

<p>References:</p>
For log Probabilites: https://www.baeldung.com/cs/naive-bayes-classification-performance
<br>
https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
