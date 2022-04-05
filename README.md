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

### 1.Finding Best Route: 
<h4> a) Base problem </h4>
<h6>Our algorithm gives us the best route which will take the shortest path out of all the available charging stations.</h6>
<img src="Images/x1_4.gif" />
<h4> b) Now with traffic data </h4>
<h6>The benefit of this grid approach instead of graph approach is that we can very easily put layers of information on top of this and our algorithm will work fine. Like we can add traffic information on top of it.</h6>
<img src="Images/x1_5.gif" />
<h4> c) With remaining battery<h4>
<h6>We can also pass the information about available batteries and it will tell us if there is any charging station within our battery limit. </h6>
<img src="Images/x1_8.gif" />


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
