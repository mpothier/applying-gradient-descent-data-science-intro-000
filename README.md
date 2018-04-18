
## Improving Regression Lines

### Introduction

In the last lesson, we derived the functions that help us descend along our cost functions efficiently.  Remember that this technique is not so different from what we saw with using the derivative to tell us our next step size and direction in two dimensions.  

![](./tangent-lines.png)

When descending along our cost curve in two dimensions, we used the slope of the tangent line at each point, to tell us how large of a step to take next.  And with the our cost curve being a function of $m$ and $b$, we had to use the gradient to determine each step.  

![](./gradientdescent.png)

But really it's an analogous approach - much like how we use the derivative of a function $f(x)$ to calculate the slope at a given value of $x$ on the graph, and thus discover our next step.  Here, we calculated the partial derivative with respect to both variables, the slope and y-intercept, to calculate the amount to move in either direction that best steers us towards our minimum RSS.   

![](./maps-pointer.png)

### Reviewing our gradient descent formulas

Luckily for us, we already did the hard work of deriving these formulas.  Now we get to see the fruits of our labor.  The following formulas tell us how to update regression variables of $m$ and $b$ to approach a "best fit" line.   

* $ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$ 
* $ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $

The formulas above tell us to take some dataset, with values of $x$ and $y$, and then given a regression formula with values $m$ and $b$, iterate through our dataset, and use the formulas to calculate updates for $m$ and $b$.  Ultimately, to descend along the cost function, we will use the following calculations:

`current_m` = `old_m` $ -  (-2*\sum_{i=1}^n x_i*\epsilon_i )$

`current_b` =  `old_b` $ - ( -2*\sum_{i=1}^n \epsilon_i )$

Time to translate these formulas into code.  First, let's initialize some data.


```python
# our data
first_show = {'x': 30, 'y': 45}
second_show = {'x': 40, 'y': 60}
third_show = {'x': 100, 'y': 150}

shows = [first_show, second_show, third_show]
```

We will create an initial regression line with the $m$ and $b$ variables set to zero so that the line lays horizontally along the x-axis.  To update our regression line, we iterate through each of the points in the dataset, and at each iteration change our `update_to_b` by $-2*\epsilon$ and change our `update_to_m` by $-2*x*\epsilon$.


```python
# initial variables of our regression line
b_current = 0
m_current = 0

#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0 

def error_at(point, b, m):
    return (m*point['x'] + b - point['y'])

for i in range(0, len(shows)):
    update_to_b += -2*(error_at(shows[i], b_current, m_current))
    update_to_m += -2*(error_at(shows[i], b_current, m_current)*shows[i]['x'])

new_b = b_current - update_to_b
new_m = m_current - update_to_m
```

In the last two lines of the code above, we calculate our `new_b` and `new_m` values by updating our taking our current values and adding our respective updates.  We define a function called `error_at`, which we can use in the error component of our partial derivatives above.

The code above represents just one update to our regression line, and therefore just one step towards our best fit line.  We'll repeat the process to take multiple steps, but first we need to make a few more changes. 

### Tweaking our approach 

The code above is very close to what we want, but we still need to make a few tweaks.

The first problem becomes clear when we think about what these formulas actually do. As of now, we are changing the $m$ and $b$ variables by at least the sum of all of the errors based upon our line's predictions relative to our actual data.  Our line would change drastically at every iteration through the formulas.  We need to apply a learning rate to each of these partial derivates to ensure that we avoid drastically updating our regression line with each step.  As we have seen before, the learning rate is just a small number, like $.0001$ which controls the amount with which we update the regression line.  The learning rate is  represented by the Greek letters eta, $\eta$, or alpha $\alpha$.  We'll use eta, so $\eta = .0001$ means that our learning rate is $.0001$.

Rememver that the gradient,  $ \nabla J(m,b)$, steers us in the correct direction.  In other words, our derivatives ensure we are making the correct **proportional** changes to $m$ and $b$.  Applying a learning rate to scale down changes to our regression line works fine, as long as the proportion and direction of the move remains the same.  While were at it, we need not multiply our partials by 2.  Again, we're in good shape so long as our changes are proportional.

![](./regression-scatter.png)

For our second tweak, note that generally when we have a larger dataset, we also will have a larger sum of errors.  That doesn't mean our formulas are less accurate, and therefore deserve larger changes.  It just means that the total error is larger.  Rather, we should think of accuracy in relation to the size of our dataset.  We can correct for this effect by dividing our update by the size of our dataset, $n$.

Making these changes, our formula looks like the following:


```python
#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0 

learning_rate = .0001
n = len(shows)
for i in range(0, n):
    
    update_to_b += -(1/n)*(error_at(shows[i], b_current, m_current))
    update_to_m += -(1/n)*(error_at(shows[i], b_current, m_current)*shows[i]['x'])

new_b = b_current - (learning_rate*update_to_b)
new_m = m_current - (learning_rate*update_to_m)
```

Our code now reflects what we know about the gradient descent process.  We start with an initial regression line with arbitrary values for $m$ and $b$.  Then for each point, we calculate how the regression line fares against the actual point (that is, we find the error).  We update what our next step to the respective variable should be by using the partial derivative.  After iterating through all of the points, we update the value of $b$ and $m$ appropriately, scaled down by a learning rate.

### Seeing our gradient descent formulas in action

As mentioned earlier, the code above represents just one update to our regression line, and therefore just one step towards our best fit line.  To take multiple steps, we wrap the process we want to repeat in a function so we can call it as much as we want.  We'll call this function `step_gradient`. 


```python
first_show = {'x': 30, 'y': 45}
second_show = {'x': 40, 'y': 60}
third_show = {'x': 100, 'y': 150}

shows = [first_show, second_show, third_show]

def step_gradient(b_current, m_current, points):
    b_gradient = 0
    m_gradient = 0
    learning_rate = .0001
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i]['x']
        y = points[i]['y']
        b_gradient += -(1/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(1/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return {'b': new_b, 'm': new_m}
```


```python
b = 0
m = 0

step_gradient(b, m, shows) # {'b': 0.0085, 'm': 0.6249999999999999}
```

We begin by setting both $b$ and $m$ to $0$.  Our `step_gradient` function calculates new $b$ and $m$ values of .0085 and .6245, respectively.  Now we need to take another step in the correct direction by calling the `step_gradient` function with these updated $b$ and $m$ values.


```python
updated_b = 0.0085
updated_m = 0.6249
step_gradient(updated_b, updated_m, shows) # {'b': 0.01345805, 'm': 0.9894768333333332}
```

Let's do this, say, 10 times.


```python
# set our initial step with m and b values, and the corresponding error.
b = 0
m = 0
iterations = []
for i in range(10):
    iteration = step_gradient(b, m, shows)
    # {'b': value, 'm': value}
    b = iteration['b']
    m = iteration['m']
    # update values of b and m
    iterations.append(iteration)
```

Let's take a look at these iterations.


```python
iterations
```

As you can see, our $m$ and $b$ values both update with each step.  Notice that with each step, the size of the changes to $m$ and $b$ decrease because they are approaching a best fit line.

###  Animating these changes

We can use Plotly to visualize these changes to our regression line.  We'll write a method called `to_line` that takes a dictionary of $m$ and $b$ variables and produces a line object for each pair.  We can then see our line changes over time. 


```python
def to_line(m, b):
    initial_x = 0
    ending_x = 100
    initial_y = m*initial_x + b
    ending_y = m*ending_x + b
    return {'data': [{'x': [initial_x, ending_x], 'y': [initial_y, ending_y]}]}

frames = list(map(lambda iteration: to_line(iteration['m'], iteration['b']),iterations))
frames[0]
```

Now we can see how our regression line changes, and approaches our data, with each iteration.


```python
from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML

init_notebook_mode(connected=True)

x_values_of_shows = list(map(lambda show: show['x'], shows))
y_values_of_shows = list(map(lambda show: show['y'], shows))
figure = {'data': [{'x': [0], 'y': [0]}, {'x': x_values_of_shows, 'y': y_values_of_shows, 'mode': 'markers'}],
          'layout': {'xaxis': {'range': [0, 110], 'autorange': False},
                     'yaxis': {'range': [0,160], 'autorange': False},
                     'title': 'Regression Line',
                     'updatemenus': [{'type': 'buttons',
                                      'buttons': [{'label': 'Play',
                                                   'method': 'animate',
                                                   'args': [None]}]}]
                    },
          'frames': frames}
iplot(figure)
```

As you can see, our regression line starts off far away from our best fit.  But it uses our `step_gradient` function to move closer to finding the line that produces the lowest error.

### Summary

In this section, we saw our gradient descent formulas in action.  The core of the gradient descent functions are understanding the two lines: 

$$ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$$
$$ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $$
    
Which both look to the errors of the current regression line for our dataset to determine how to update the regression line next.  These formulas came from our cost function, $J(m,b) = \sum_{i = 1}^n(y_i - (mx_i + b))^2 $, and using the gradient to find the direction of steepest descent.  We translated theses formulas into code so we could visualize our regression line as it continued to improve its alignment with the data.  
