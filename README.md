Download Link: https://assignmentchef.com/product/solved-col774-assignment-1
<br>
<h1>1.     Linear Regression</h1>

In this problem, we will implement least squares linear regression to predict density of wine based on its acidity. Recall that the error metric for least squares is given by:

where <em>h<sub>θ</sub></em>(<em>x</em>) = <em>θ<sup>T</sup>x </em>and all the symbols are as discussed in the class. The files linearX.csv and linearY.csv contain the acidity of the wine (<em>x</em><sup>(<em>i</em>)</sup>’s, <em>x</em><sup>(<em>i</em>) </sup>∈ R) and its density (<em>y</em><sup>(<em>i</em>)</sup>’s, <em>y</em><sup>(<em>i</em>) </sup>∈ R), respectively, with one training example per row. We will implement least squares linear regression to learn the relationship between <em>x</em><sup>(<em>i</em>)</sup>’s and <em>y</em><sup>(<em>i</em>)</sup>’s.

<ul>

 <li> Implement batch gradient descent method for optimizing <em>J</em>(<em>θ</em>). Choose an appropriate learning rate and the stopping criteria (as a function of the change in the value of <em>J</em>(<em>θ</em>)). You can initialize the parameters as <em>θ </em>= <em>~</em>0 (the vector of all zeros). Do not forget to include the intercept term. Report your learning rate, stopping criteria and the final set of parameters obtained by your algorithm.</li>

 <li> Plot the data on a two-dimensional graph and plot the hypothesis function learned by your algorithm in the previous part.</li>

 <li> Draw a 3-dimensional mesh showing the error function (<em>J</em>(<em>θ</em>)) on <em>z</em>-axis and the parameters in the <em>x </em>− <em>y </em> Display the error value using the current set of parameters at each iteration of the gradient descent. Include a time gap of 0<em>.</em>2 seconds in your display for each iteration so that the change in the function value can be observed by the human eye.</li>

 <li>Repeat the part above for drawing the contours of the error function at each iteration of the gradient descent. Once again, chose a time gap of 0<em>.</em>2 seconds so that the change be perceived by the human eye.(Note here plot will be 2-D)</li>

 <li>Repeat the part above (i.e. draw the contours at each learning iteration) for the step size values of <em>η </em>= {0<em>.</em>001<em>,</em>0<em>.</em>025<em>,</em>0<em>.</em>1}. What do you observe? Comment.</li>

</ul>

<h1>2.  Sampling and Stochastic Gradient Descent</h1>

In this problem, we will introduce the idea of sampling by adding Gaussian noise to the prediction of a hypothesis and generate synthetic training data. Consider a given hypothesis <em>h<sub>θ </sub></em>(<em>i.e. </em>known <em>θ</em><sub>0</sub><em>,θ</em><sub>1</sub><em>,θ</em><sub>2</sub>) for

a data point. Note that <em>x</em><sub>0 </sub>= 1 is the intercept term.

<em>y </em>= <em>h<sub>θ</sub></em>(<em>x</em>) = <em>θ</em><sub>0 </sub>+ <em>θ</em><sub>1</sub><em>x</em><sub>1 </sub>+ <em>θ</em><sub>2</sub><em>x</em><sub>2 </sub>Adding Gaussian noise, equation becomes

where

To gain deeper understanding behind Stochastic Gradient Descent (SGD), we will use the SGD algorithm to learn the original hypothesis from the data generated using sampling, for varying batch sizes. We will implement the version where we make a complete pass through the data in a round robin fashion (after initially shuffling the examples). If there are <em>r </em>examples in each batch, then there is a total of batches assuming <em>m </em>training examples. For the batch number), the set of examples is given as:

{<em>x</em><sup>(<em>i</em></sup><sup>1)</sup><em>,x</em><sup>(<em>i</em></sup><sup>2)</sup><em>,</em>··· <em>,x</em><sup>(<em>i</em></sup><em><sup>r</sup></em><sup>)</sup>} where <em>i<sub>k </sub></em>= (<em>b</em>−1)<em>r </em>+<em>k</em>. The Loss function computed over these <em>r </em>examples is given as:

<ul>

 <li><strong> </strong>Sample 1 million data points taking values of 4) and <em>x</em><sub>2 </sub>∼</li>

</ul>

N(−1<em>,</em>4) independently, and noise variance in <em>y</em>, <em>σ</em><sup>2 </sup>= 2.

<ul>

 <li><strong> </strong>Implement Stochastic gradient descent method for optimizing <em>J</em>(<em>θ</em>). Relearn</li>

</ul>

using sampled data points of part a) keeping everything same except the batch size. Keep <em>η </em>= 0<em>.</em>001 and initialize ∀<em>j θ<sub>j </sub></em>= 0. Report the <em>θ </em>learned each time separately for values of batch size <em>r </em>= {1<em>,</em>100<em>,</em>10000<em>,</em>1000000}. Carefully decide your convergence criteria in each case. Make sure to watch the online video posted on the course website for deciding the convergence of SGD algorithm.

<ul>

 <li><strong> </strong>Do different algorithms in the part above (for varying values of <em>r</em>) converge to the same parameter values? How much different are these from the parameters of the original hypothesis from which the data was generated? Comment on the relative speed of convergence and also on number of iterations in each case. Next, for each of learned models above, report the error on a new test data of 10,000 samples provided in the file named csv. Note that this test set was generated using the same sampling procedure as described in part (a) above. Also, compute the test error with respect to the prediction of the original hypothesis, and compare with the error obtained using learned hypothesis in each case. Comment.</li>

 <li>In the 3 dimensional parameter space(<em>θ<sub>j </sub></em>on each axis), plot the movement of <em>θ </em>as the parameters are updated (until convergence) for varying batch sizes. How does the (shape of) movement compare in each case? Does it make intuitive sense? Argue.</li>

</ul>

<h1>3.    Logistic Regression</h1>

Consider the log-likelihood function for logistic regression:

For the following, you will need to calculate the value of the Hessian H of the above function.

<ul>

 <li><strong>(10 points) </strong>The files csv and logisticY.csv contain the inputs (<em>x</em><sup>(<em>i</em>) </sup>∈ <em>R</em><sup>2</sup>) and outputs (<em>y</em><sup>(<em>i</em>) </sup>∈ {0<em>,</em>1}) respectively for a binary classification problem, with one training example per row. Implement<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> Newton’s method for optimizing <em>L</em>(<em>θ</em>), and apply it to fit a logistic regression model to the data. Initialize Newton’s method with <em>θ </em>= <em>~</em>0 (the vector of all zeros). What are the coefficients <em>θ </em>resulting from your fit? (Remember to include the intercept term.)</li>

 <li><strong>(5 points) </strong>Plot the training data (your axes should be <em>x</em><sub>1 </sub>and <em>x</em><sub>2 </sub>, corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (i.e., this should be a straight line showing the boundary separating the region where <em>h</em>(<em>x</em>) <em>&gt; </em>0<em>.</em>5 from where <em>h</em>(<em>x</em>) ≤ 0<em>.</em>)</li>

</ul>

<h1>4.     Gaussian Discrmimant Analysis</h1>

In this problem, we will implement GDA for separating out salmons from Alaska and Canada. Each salmon is represented by two attributes <em>x</em><sub>1 </sub>and <em>x</em><sub>2 </sub>depicting growth ring diameters in 1) fresh water, 2) marine water, respectively. File q4x.dat stores the two attribute values with one entry on each row. File q4y.dat contains the target values (<em>y</em><sup>(<em>i</em>)</sup>’s ∈ {Alaska, Canada}) on respective rows.

<ul>

 <li><strong> </strong>Implement Gaussian Discriminant Analysis using the closed form equations described in class. Assume that both the classes have the same co-variance matrix i.e. Σ<sub>0 </sub>= Σ<sub>1 </sub>= Σ. Report the values of the means, <em>µ</em><sub>0 </sub>and <em>µ</em><sub>1</sub>, and the co-variance matrix Σ.</li>

 <li>Plot the training data corresponding to the two coordinates of the input features, and you should use a different symbol for each point plotted to indicate whether that example had label Canada or Alaska.</li>

 <li>Describe the equation of the boundary separating the two regions in terms of the parameters <em>µ</em><sub>0</sub><em>,µ</em><sub>1 </sub>and Σ. Recall that GDA results in a linear separator when the two classes have identical covariance matrix. Along with the data points plotted in the part above, plot (on the same figure) decision boundary fit by GDA.</li>

 <li>In general, GDA allows each of the target classes to have its own covariance matrix. This results (in general) results in a quadratic boundary separating the two class regions. In this case, the maximum-likelihood estimate of the co-variance matrix Σ<sub>0 </sub>can be derived using the equation:</li>

</ul>

(1)

And similarly, for Σ<sub>1</sub>. The expressions for the means remain the same as before. Implement GDA for the above problem in this more general setting. Report the values of the parameter estimates i.e. <em>µ</em><sub>0</sub>, <em>µ</em><sub>1</sub>, Σ<sub>0</sub>, Σ<sub>1</sub>.

<ul>

 <li><strong> </strong>Describe the equation for the quadratic boundary separating the two regions in terms of the parameters <em>µ</em><sub>0</sub>, <em>µ</em><sub>1 </sub>and Σ<sub>0</sub>, Σ<sub>1</sub>. On the graph plotted earlier displaying the data points and the linear separating boundary, also plot the quadratic boundary obtained in the previous step.</li>

 <li><strong> </strong>Carefully analyze the linear as well as the quadratic boundaries obtained. Comment on your observations.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> Write your own version, and do not call a built-in library function.