import matplotlib.pyplot as plt
from matplotlib.patches import Arc
def createPitch(length,width, unity,linecolor): # in meters
# Code by @JPJ_dejong
"""
creates a plot in which the 'length' is the length of the pitch (goal to goal).
And 'width' is the width of the pitch (sideline to sideline).
Fill in the unity in meters or in yards.
"""
#Set unity
if unity == "meters":
# Set boundaries
if length >= 120.5 or width >= 75.5:
return(str("Field dimensions are too big for meters as unity, didn't you mean yards as
unity?\
Otherwise the maximum length is 120 meters and the maximum width is 75
meters. Please try again"))
#Run program if unity and boundaries are accepted
else:
#Create figure
fig=plt.figure()
#fig.set_size_inches(7, 5)
23
ax=fig.add_subplot(1,1,1)
#Pitch Outline & Centre Line
plt.plot([0,0],[0,width], color=linecolor)
plt.plot([0,length],[width,width], color=linecolor)
plt.plot([length,length],[width,0], color=linecolor)
plt.plot([length,0],[0,0], color=linecolor)
plt.plot([length/2,length/2],[0,width], color=linecolor)
#Left Penalty Area
plt.plot([16.5 ,16.5],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
plt.plot([0,16.5],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
plt.plot([16.5,0],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
#Right Penalty Area
plt.plot([(length-16.5),length],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
plt.plot([(length-16.5), (length-16.5)],[(width/2
+16.5),(width/2-16.5)],color=linecolor)
plt.plot([(length-16.5),length],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
#Left 5-meters Box
plt.plot([0,5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
plt.plot([5.5,5.5],[(width/2+7.32/2+5.5),(width/2-7.32/2-5.5)],color=linecolor)
plt.plot([5.5,0.5],[(width/2-7.32/2-5.5),(width/2-7.32/2-5.5)],color=linecolor)
#Right 5 -eters Box
plt.plot([length,length-5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
plt.plot([length-5.5,length-5.5],[(width/2+7.32/2+5.5),width/2-7.32/2-5.5],color=linecolor)
plt.plot([length-5.5,length],[width/2-7.32/2-5.5,width/2-7.32/2-5.5],color=linecolor)
#Prepare Circles
24
centreCircle = plt.Circle((length/2,width/2),9.15,color=linecolor,fill=False)
centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
leftPenSpot = plt.Circle((11,width/2),0.8,color=linecolor)
rightPenSpot = plt.Circle((length-11,width/2),0.8,color=linecolor)
#Draw Circles
ax.add_patch(centreCircle)
ax.add_patch(centreSpot)
ax.add_patch(leftPenSpot)
ax.add_patch(rightPenSpot)
#Prepare Arcs
leftArc =
Arc((11,width/2),height=18.3,width=18.3,angle=0,theta1=308,theta2=52,color=linecolor)
rightArc =
Arc((length-11,width/2),height=18.3,width=18.3,angle=0,theta1=128,theta2=232,color=linec
olor)
#Draw Arcs
ax.add_patch(leftArc)
ax.add_patch(rightArc)
#Axis titles
#check unity again
elif unity == "yards":
#check boundaries again
if length <= 95:
return(str("Didn't you mean meters as unity?"))
elif length >= 131 or width >= 101:
return(str("Field dimensions are too big. Maximum length is 130, maximum width is
100"))
else:
#Run program if unity and boundaries are accepted
25
#Create figure
fig=plt.figure()
#fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
#Pitch Outline & Centre Line
plt.plot([0,0],[0,width], color=linecolor)
plt.plot([0,length],[width,width], color=linecolor)
plt.plot([length,length],[width,0], color=linecolor)
plt.plot([length,0],[0,0], color=linecolor)
plt.plot([length/2,length/2],[0,width], color=linecolor)
#Left Penalty Area
plt.plot([18 ,18],[(width/2 +18),(width/2-18)],color=linecolor)
plt.plot([0,18],[(width/2 +18),(width/2 +18)],color=linecolor)
plt.plot([18,0],[(width/2 -18),(width/2 -18)],color=linecolor)
#Right Penalty Area
plt.plot([(length-18),length],[(width/2 +18),(width/2 +18)],color=linecolor)
plt.plot([(length-18), (length-18)],[(width/2 +18),(width/2-18)],color=linecolor)
plt.plot([(length-18),length],[(width/2 -18),(width/2 -18)],color=linecolor)
#Left 6-yard Box
plt.plot([0,6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
plt.plot([6,6],[(width/2+7.32/2+6),(width/2-7.32/2-6)],color=linecolor)
plt.plot([6,0],[(width/2-7.32/2-6),(width/2-7.32/2-6)],color=linecolor)
#Right 6-yard Box
plt.plot([length,length-6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
plt.plot([length-6,length-6],[(width/2+7.32/2+6),width/2-7.32/2-6],color=linecolor)
plt.plot([length-6,length],[(width/2-7.32/2-6),width/2-7.32/2-6],color=linecolor)
#Prepare Circles; 10 yards distance. penalty on 12 yards
26
centreCircle = plt.Circle((length/2,width/2),10,color=linecolor,fill=False)
centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
leftPenSpot = plt.Circle((12,width/2),0.8,color=linecolor)
rightPenSpot = plt.Circle((length-12,width/2),0.8,color=linecolor)
#Draw Circles
ax.add_patch(centreCircle)
ax.add_patch(centreSpot)
ax.add_patch(leftPenSpot)
ax.add_patch(rightPenSpot)
#Prepare Arcs
leftArc =
Arc((11,width/2),height=20,width=20,angle=0,theta1=312,theta2=48,color=linecolor)
rightArc =
Arc((length-11,width/2),height=20,width=20,angle=0,theta1=130,theta2=230,color=linecolor
)
#Draw Arcs
ax.add_patch(leftArc)
ax.add_patch(rightArc)
#Tidy Axes
plt.axis('off')
return fig,ax
def createPitchOld():
#Taken from FC Python
#Create figure
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
27
linecolor='black'
#Pitch Outline & Centre Line
plt.plot([0,0],[0,90], color=linecolor)
plt.plot([0,130],[90,90], color=linecolor)
plt.plot([130,130],[90,0], color=linecolor)
plt.plot([130,0],[0,0], color=linecolor)
plt.plot([65,65],[0,90], color=linecolor)
#Left Penalty Area
plt.plot([16.5,16.5],[65,25],color=linecolor)
plt.plot([0,16.5],[65,65],color=linecolor)
plt.plot([16.5,0],[25,25],color=linecolor)
#Right Penalty Area
plt.plot([130,113.5],[65,65],color=linecolor)
plt.plot([113.5,113.5],[65,25],color=linecolor)
plt.plot([113.5,130],[25,25],color=linecolor)
#Left 6-yard Box
plt.plot([0,5.5],[54,54],color=linecolor)
plt.plot([5.5,5.5],[54,36],color=linecolor)
plt.plot([5.5,0.5],[36,36],color=linecolor)
#Right 6-yard Box
plt.plot([130,124.5],[54,54],color=linecolor)
plt.plot([124.5,124.5],[54,36],color=linecolor)
plt.plot([124.5,130],[36,36],color=linecolor)
#Prepare Circles
centreCircle = plt.Circle((65,45),9.15,color=linecolor,fill=False)
centreSpot = plt.Circle((65,45),0.8,color=linecolor)
leftPenSpot = plt.Circle((11,45),0.8,color=linecolor)
rightPenSpot = plt.Circle((119,45),0.8,color=linecolor)
28
#Draw Circles
ax.add_patch(centreCircle)
ax.add_patch(centreSpot)
ax.add_patch(leftPenSpot)
ax.add_patch(rightPenSpot)
#Prepare Arcs
leftArc =
Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color=linecolor)
rightArc =
Arc((119,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color=linecolor)
#Draw Arcs
ax.add_patch(leftArc)
ax.add_patch(rightArc)
#Tidy Axes
plt.axis('off')
return fig,ax
def createGoalMouth():
#Adopted from FC Python
#Create figure
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
linecolor='black'
#Pitch Outline & Centre Line
plt.plot([0,65],[0,0], color=linecolor)
plt.plot([65,65],[50,0], color=linecolor)
29
plt.plot([0,0],[50,0], color=linecolor)
#Left Penalty Area
plt.plot([12.5,52.5],[16.5,16.5],color=linecolor)
plt.plot([52.5,52.5],[16.5,0],color=linecolor)
plt.plot([12.5,12.5],[0,16.5],color=linecolor)
#Left 6-yard Box
plt.plot([41.5,41.5],[5.5,0],color=linecolor)
plt.plot([23.5,41.5],[5.5,5.5],color=linecolor)
plt.plot([23.5,23.5],[0,5.5],color=linecolor)
#Goal
plt.plot([41.5-5.34,41.5-5.34],[-2,0],color=linecolor)
plt.plot([23.5+5.34,41.5-5.34],[-2,-2],color=linecolor)
plt.plot([23.5+5.34,23.5+5.34],[0,-2],color=linecolor)
#Prepare Circles
leftPenSpot = plt.Circle((65/2,11),0.8,color=linecolor)
#Draw Circles
ax.add_patch(leftPenSpot)
#Prepare Arcs
leftArc =
Arc((32.5,11),height=18.3,width=18.3,angle=0,theta1=38,theta2=142,color=linecolor)
#Draw Arcs
ax.add_patch(leftArc)
#Tidy Axes
plt.axis('off')
30
return fig,ax
"""
Spyder Editor
This is a temporary script file.
"""
# The basics
import pandas as pd
import numpy as np
import json
# Plotting
# Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Decide which league to load
# Wyscout data from
https://figshare.com/collections/Soccer_match_event_dataset/4415000/2
with open('/Users/theo/fcprj/events/events_England.json') as f:
data = json.load(f)
# Create a data set of shots.
train = pd.DataFrame(data)
pd.unique(train['subEventName'])
shots = train[train['subEventName'] == 'Shot']
shots_model = pd.DataFrame(columns=['Goal', 'X', 'Y'])
# Go through the dataframe and calculate X, Y co-ordinates.
# Distance from a line in the centre
# Shot angle.
31
# Details of tags can be found here: https://apidocs.wyscout.com/matches-wyid-events
for i, shot in shots.iterrows():
header = 0
for shottags in shot['tags']:
if shottags['id'] == 403:
header = 1
# Only include non-headers
if not (header):
shots_model.at[i, 'X'] = 100 - shot['positions'][0]['x']
shots_model.at[i, 'Y'] = shot['positions'][0]['y']
shots_model.at[i, 'C'] = abs(shot['positions'][0]['y'] - 50)
# Distance in metres and shot angle in radians.
x = shots_model.at[i, 'X'] * 105 / 100
y = shots_model.at[i, 'C'] * 65 / 100
shots_model.at[i, 'Distance'] = np.sqrt(x ** 2 + y ** 2)
a = np.arctan(7.32 * x / (x ** 2 + y ** 2 - (7.32 / 2) ** 2))
if a < 0:
a = np.pi + a
shots_model.at[i, 'Angle'] = a
# Was it a goal
shots_model.at[i, 'Goal'] = 0
for shottags in shot['tags']:
# Tags contain that its a goal
if shottags['id'] == 101:
shots_model.at[i, 'Goal'] = 1
# Two dimensional histogram
H_Shot = np.histogram2d(shots_model['X'], shots_model['Y'], bins=50, range=[[0, 100], [0,
100]])
goals_only = shots_model[shots_model['Goal'] == 1]
32
H_Goal = np.histogram2d(goals_only['X'], goals_only['Y'], bins=50, range=[[0, 100], [0,
100]])
# Plot the number of shots from different points
(fig, ax) = createGoalMouth()
pos = ax.imshow(H_Shot[0], extent=[-1, 66, 104, -1], aspect='auto', cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of shots')
plt.xlim((-1, 66))
plt.ylim((-3, 35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig('NumberOfShots.png', dpi=None, bbox_inches="tight")
# Plot the number of GOALS from different points
(fig, ax) = createGoalMouth()
pos = ax.imshow(H_Goal[0], extent=[-1, 66, 104, -1], aspect='auto', cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of goals')
plt.xlim((-1, 66))
plt.ylim((-3, 35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig('NumberOfGoals.png', dpi=None, bbox_inches="tight")
# Plot the probability of scoring from different points
(fig, ax) = createGoalMouth()
pos = ax.imshow(H_Goal[0] / H_Shot[0], extent=[-1, 66, 104, -1], aspect='auto',
cmap=plt.cm.Reds, vmin=0, vmax=0.5)
fig.colorbar(pos, ax=ax)
ax.set_title('Proportion of shots resulting in a goal')
33
plt.xlim((-1, 66))
plt.ylim((-3, 35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig('ProbabilityOfScoring.pdf', dpi=None, bbox_inches="tight")
#Plot a logistic curve
b=[3, -3]
x=np.arange(5,step=0.1)
y=1/(1+np.exp(-b[0]-b[1]*x))
fig,ax=plt.subplots(num=1)
plt.ylim((-0.05,1.05))
plt.xlim((0,5))
ax.set_ylabel('y')
ax.set_xlabel("x")
ax.plot(x, y, linestyle='solid', color='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
#Get first 200 shots
shots_200=shots_model.iloc[:200]
#Plot first 200 shots goal angle
fig,ax=plt.subplots(num=1)
ax.plot(shots_200['Angle']*180/np.pi, shots_200['Goal'], linestyle='none', marker= '.',
markersize= 10.0, color='black')
ax.set_ylabel('Goal scored')
ax.set_xlabel("Shot angle (degrees)")
plt.ylim((-0.05,1.05))
ax.set_yticks([0,1])
34
ax.set_yticklabels(['No','Yes'])
plt.show()
#Show empirically how goal angle predicts probability of scoring
shotcount_dist=np.histogram(shots_model['Angle']*180/np.pi,bins=40,range=[0, 150])
goalcount_dist=np.histogram(goals_only['Angle']*180/np.pi,bins=40,range=[0, 150])
prob_goal=np.divide(goalcount_dist[0],shotcount_dist[0])
angle=shotcount_dist[1]
midangle= (angle[:-1] + angle[1:])/2
fig,ax=plt.subplots(num=2)
ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Shot angle (degrees)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
b=[3, -3]
x=np.arange(150,step=0.1)
y=1/(1+np.exp(b[0]+b[1]*x*np.pi/180))
ax.plot(x, y, linestyle='solid', color='black')
plt.show()
xG=1/(1+np.exp(b[0]+b[1]*shots_model['Angle']))
shots_model = shots_model.assign(xG=xG)
shots_200=shots_model.iloc[:200]
fig,ax=plt.subplots(num=1)
ax.plot(shots_200['Angle']*180/np.pi, shots_200['Goal'], linestyle='none', marker= '.',
markersize= 12, color='black')
ax.plot(x, y, linestyle='solid', color='black')
ax.plot(x, 1-y, linestyle='solid', color='black')
loglikelihood=0
for item,shot in shots_200.iterrows():
ang=shot['Angle']*180/np.pi
35
if shot['Goal']==1:
loglikelihood=loglikelihood+np.log(shot['xG'])
ax.plot([ang,ang],[shot['Goal'],shot['xG']], color='red')
else:
loglikelihood=loglikelihood+np.log(1 - shot['xG'])
ax.plot([ang,ang],[shot['Goal'],1-shot['xG']], color='blue')
ax.set_ylabel('Goal scored')
ax.set_xlabel("Shot angle (degrees)")
plt.ylim((-0.05,1.05))
plt.xlim((0,80))
#plt.text(45,0.2,'Log-likelihood:')
#plt.text(45,0.1,str(loglikelihood))
ax.set_yticks([0,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('LikelihoodExample.pdf', dpi=None, bbox_inches="tight")
plt.show()
#Make single variable model of angle
#Using logistic regression we find the optimal values of b
#This process minimizes the loglikelihood
test_model = smf.glm(formula="Goal ~ Angle" , data=shots_model,
family=sm.families.Binomial()).fit()
print(test_model.summary())
b=test_model.params
xGprob=1/(1+np.exp(b[0]+b[1]*midangle*np.pi/180))
fig,ax=plt.subplots(num=1)
ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
ax.plot(midangle, xGprob, linestyle='solid', color='black')
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Shot angle (degrees)")
ax.spines['top'].set_visible(False)
36
ax.spines['right'].set_visible(False)
plt.show()
fig.savefig('ProbabilityOfScoringAngleFit.pdf', dpi=None, bbox_inches="tight")
#Show empirically how distance from goal predicts probability of scoring
shotcount_dist=np.histogram(shots_model['Distance'],bins=40,range=[0, 70])
goalcount_dist=np.histogram(goals_only['Distance'],bins=40,range=[0, 70])
prob_goal=np.divide(goalcount_dist[0],shotcount_dist[0])
distance=shotcount_dist[1]
middistance= (distance[:-1] + distance[1:])/2
fig,ax=plt.subplots(num=1)
ax.plot(middistance, prob_goal, linestyle='none', marker= '.', color='black')
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Distance from goal (metres)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#Make single variable model of distance
test_model = smf.glm(formula="Goal ~ Distance" , data=shots_model,
family=sm.families.Binomial()).fit()
print(test_model.summary())
b=test_model.params
xGprob=1/(1+np.exp(b[0]+b[1]*middistance))
ax.plot(middistance, xGprob, linestyle='solid', color='black')
plt.show()
fig.savefig('Output/ProbabilityOfScoringDistance.pdf', dpi=None, bbox_inches="tight")
#Adding distance squared
squaredD = shots_model['Distance']**2
shots_model = shots_model.assign(D2=squaredD)
test_model = smf.glm(formula="Goal ~ Distance + D2" , data=shots_model,
family=sm.families.Binomial()).fit()
print(test_model.summary())
b=test_model.params
37
xGprob=1/(1+np.exp(b[0]+b[1]*middistance+b[2]*pow(middistance,2)))
fig,ax=plt.subplots(num=1)
ax.plot(middistance, prob_goal, linestyle='none', marker= '.', color='black')
ax.set_ylabel('Probability chance scored')
ax.set_xlabel("Distance from goal (metres)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(middistance, xGprob, linestyle='solid', color='black')
plt.show()
fig.savefig('/ProbabilityOfScoringDistanceSquared.pdf', dpi=None, bbox_inches="tight")
#Adding even more variables to the model.
squaredX = shots_model['X']**2
shots_model = shots_model.assign(X2=squaredX)
squaredC = shots_model['C']**2
shots_model = shots_model.assign(C2=squaredC)
AX = shots_model['Angle']*shots_model['X']
shots_model = shots_model.assign(AX=AX)
