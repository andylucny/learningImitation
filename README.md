# learningImitation
one-shot learning of the imitation task

The directory whistle contains the test of the whistling detection. 

The directory dino implements test of the medium size DINO backbone that turns image into 384 features.

The directory pyicubsim is the client library of the iCubSim (the simulator of the humanoid robot iCub).
It implements communication with the simulator and the direct and inverse kinematics.

The directory kinematics employs the kinematics for creation of the dataset of the reasonable iCub arm poses.
kinematics_demo.py tests that kinematics works. renew.py sets the robot into the standard position.
kinematics_arm.py generates the dataset.

The directory vae trains the autoencoder of the robot actions and distillates the decoder part.
create_dataset.py generates two arms joints positions from the kinematics dataset.
vae_icub.py trains their variational autoencoder and distillates the decoder part.
keras2pb.py converts the decoder from the keras format to the pb suitable for the OpenCV library.
gener.py is a demo written in pure OpenCV that employs the decoder model for robot actions generation.

The directory integration contains implementation of the real-time system that performs the imitation game.
It is based on two phases: invitation and imitation. In the invitation phase, the person in front of camera imitates the robot.
The person copies the robot pose and whistles. After several postures, 
the person whistles for a longer time and the imitation phase starts.
The system learns in the one-shot way though it does contain the posture representation of the seen body, it just associates features provided by the backbone to the feature vectors of the robot's actions.
The operation is not sensitive on e.g. dress of the person or the color of the wall since the association is provided the Attention mechanism. 
That provides also general ability to imitate somehow any person posture. Of course, the quality is good for the shown poses only or poses that can be mixed from the shown ones.
Integration is provided by the blackboard architecture in agentspace.py.

teaser video: https://youtu.be/-3BVbU9BeRE (fair imitation) https://youtu.be/_CBnCOnWRdY (cheating)

Citation: Lucny Andrej: Towards one-shot Learning via Attention. Workshop on the Natural Computing. ITAT 2022, Zuberec, Slovakia.
https://ceur-ws.org/Vol-3226/invited4.pdf

Andrej Lucny, Comenius University, Bratislava 2022
