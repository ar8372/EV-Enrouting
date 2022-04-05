# EV-Enrouting
<h3>Find optimal path for electric vehicle using spatial information.</h3>

<h2>Preprocessing :-</h1>
<p></p>
<img src="x1_2.gif" />

## Front End of Video Talker: 
 * url: https://ms-engage-videchat-frontend.herokuapp.com/
 * backend url: https://ms-engage-video-chat-backend.herokuapp.com/
 * backend github link: https://github.com/aishik-rakshit/MS_engage_videochat_server
 * Video Demo YOutube link: https://www.youtube.com/watch?v=iGKmA1fG5Lg&t=2s&ab_channel=AishikRakshit


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
