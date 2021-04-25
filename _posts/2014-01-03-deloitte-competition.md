# 60 Day Journey of Deloitte Churn Prediction Competition

![](/images/20140103-deloitte-competition.png)

1. TOC
{:toc}

# Competition

Last December, I teamed up with Michael once again to participate in the Deloitte Churn Prediction competition at Kaggle, where to predict which customers will leave an insurance company in the next 12 months.

It was a master competition, which is open to only master level Kagglers (top 0.2% out of 138K competitors), with \$70,000 cash prizes for top 3 finishers.

# Result

We managed to do well and finished in 4th place out of 37 teams in spite of that we did not have much time due to projects at work and family events (especially for Michael, who became a dad during the competition).

Although we were little short to earn the prize, it was a fun experience working together with Michael, competing with other top competitors across the world, and climbing the leaderboard day by day.

# Visualization

I visualized our 60 day journey during the competition below, and here are some highlights (for us):

* Day 22-35: Dived into the competition, set up the github repo and S3 for collaboration, and climbed up the leaderboard quickly.
* Day 41-45: Second spurt.  Dug in GBM and NN models.  Michael’s baby girl was born on Day 48.
* Day 53-60: Last spurt.  Ensembled all models.  Improved our score every day, but didn’t have time to train the best models.

[Motion Chart - Deloitte Churn Prediction Leaderboard](/deloitte-leaderboard.html)

Once clicked the link above, it will show a motion chart where:

* X-axis: Competition day.  From day 0 to day 60.
* Y-axis: AUC score.
* Colored circle: Each team.  If clicked, it shows which team it represents.
* Right most legend: Competition day.  You can drag up and down the number to see the chart on a specific day.
* Initial positions of circles show the scores of their first submissions.

For the chart, I reused the code using rCharts published by Tony Hirst at github: https://github.com/psychemedia (He also wrote a tutorial on his blog about creating a motion chart using rCharts).

# Closing

We took a rain check on this, but will win next time!  🙂
