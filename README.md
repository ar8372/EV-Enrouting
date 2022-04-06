# EV-Enrouting
<h3>Find optimal path for electric vehicle using spatial information.</h3>

<h2>Preprocessing :-</h1>
<p></p>
<img src="Images/x1_2.gif" />

## Our algorithm solves 3 main problems: 
 * Finding Best Route
 * Finding Optimal Charging Station Location
 * Dealing with Overhead on Charging Stations

Below we will see demo of these:-

## 1.Finding Best Route: 
<h4> a) Base problem </h4>
<h6>Our algorithm gives us the best route which will take the shortest path out of all the available charging stations.</h6>
<img src="Images/x1_4.gif" />
<h4> b) Now with traffic data </h4>
<h6>The benefit of this grid approach over graph approach is that we can very easily put layers of information on top and our algorithm will work fine. Like here we have added traffic information on top of it.</h6>
<img src="Images/x1_5.gif" />
<h4> c) With remaining battery<h4>
<h6>We can also pass the information about available battery power and it will tell us if there is any charging station within our battery limit. We have given remaining battery of 100 minutes and now we see that many times it is not showing us any path  because it is out of range of 100 minutes. </h6>
<img src="Images/x1_8.gif" />
Let's look at details of how it is working:-
<img src="Images/x1_10.gif" />

## 2.Finding optimal Charging Stations location:
Finding the optimal location to set up a charging station is very tricky and we have to look at various factors, like where there is more demand and  which is geographically the most feasible location from all places.
### So to solve this we applied three approaches. 
 * a) Brute force approach
 * b) Sliding Window Technique
 * c) Subblocks Technique

<br>
a) In <b>brute force</b> approach we calculated total return by putting CS at each point of grid (50*50) and the point corresponding to maximum total return is the optimal point.<br>
b) In <b>sliding window</b>  we took a window of 10*10 and moved over this 50*50 matrix.<br>
c) In </b>Sub-blocks Techniqu</b> e we divided our whole grid into 4 sub grids (upper left, upper right, lower left, lower right). Then we calculated the return of the median point for each sub grid. We repeat the same for subgrid with max total return.<br>
[Optimality]<br>
Brute Force Approach >> Sliding Window Technique >> Sub-blocks method <br>
[Speed]<br>
Sub-blocks method >> Sliding Window Technique >> Brute Force Approach<br>
(So there is tradeoff between Speed and Optimality)
<h4>Sliding Window Techniqe </h4>
<img src="Images/x1_11.gif" />
<h4>Sliding Window with traffic</h4>
<img src="Images/x1_13.gif" /> <br>
note: we see that due to traffic the optimal charging station position has changed, which makes sense.

## 3.Overhead on Charging stations:
### For this I have defined two types of overhead on the charging stations. 
 * a) Dynamic overhead 
 * b) Static Overhead 
<br>
<b>Dynamic Overhead</b> tells how many cars are there in the queue, i.e. if we reach the Charging station now then after how much time we will get the turn. <br>
<b>Static Overhead</b> tells about on an average when a vehicle is plugged in for charging how much time it takes to get fully charged. <br>
Together these two help us find a charging station which is best for us at that current moment.<br>
[case1] : only Static Overhead


| Charging Station | Static Overhead  | travel time  | total time |
| :---:   | :-: | :-: | :-: |
| Cs1 | 50 | 20 | 70 |
| Cs2 | 10 | 28 | 38 |
| Cs3 | 5 | 38 | 43 |

So our algorithm will choose Cs2 >> Cs3 >> Cs1 
Let’s verify below.
<img src="Images/x1_15.gif" /> <br>

[case2] : both Dynamic and Static Overhead 

| Charging Station | Dynamic Overhead  | travel time  | remaining overhead:- max(0, Dynamic_overhead-travel_time) | Static Overhead | Total Overhead:- rem_overhead + static_overhead |
| :---:   | :-: | :-: | :-: | :-: | :-: |
| Cs1 | 40 | 20 | 20 | 50 | 70 |
| Cs2 | 48 | 28 | 20 | 10 | 30 |
| Cs3 | 27 | 38 | 0 | 0 | 5 |


So our algorithm will choose Cs3 >> Cs2 >> Cs1 
[ In this I have applied greedy search which allows us to find optimal CS without the need of calculating travel_time of each and every CS]
note: Greedy Search in our case gives optimal solution, details of this method can be found in the report
<img src="Images/x1_14.gif" /> <br>

Please open Backend URL once before opening the frontend URL as Heroku tends to freeze deployed Websites if the have no active visitors after a few hours

## Features implemented:
* Basic Video Chat between 2 Peers has been implemented using WebRTC by exchanging Web SocketID and SDP between 2 Users and Passing local stream (both Audio and Video) from one user to another.

## Additional Features Implemented
* Ability to turn on and Turn of Microphone
* Ability to turn on and Turn of Videocam
* Ability to Share Screen​
* Ability to hangup without leaving Web app​

## Group Call Features
* Ability to host multiple group calls at the same time with 3 or more people.
* Ability for the host to remove everyone from the call when he/ she stops it.
* Ability for individual participants to leave the call or join at any time.
* Ability to turn videocam and Microphone on and off while in call

## Adapt Feature
* Messaging feature has been implemented using WebRTC data channels.

## Limitations
* Can't share screen in group call.
* Messages are not persistent. Get deleted after some time.
* As Group call is based on PeerJS it faces difficulty handling larger number of participants.
* Messages Feature nor available in groupcalls. Only between individual participants.
