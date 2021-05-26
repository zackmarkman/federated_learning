# COMP3221 Federated Learning Assignment 2
# This README file details how to run the server and x clients

Run the server with "python COMP3221_FLServer.py <Port-server> <Sub-client>". Port-server should be set to 6000 and the Sub-client flag should be set to 0 or 1. 0 means M = K and the server will aggregate all client models, 1 means M = 2 and the server will only aggregate 2 client models over the population.

During startup, the server will wait 30 seconds for clients to initialise and connect via a handshake. To run a client, open a new terminal tab and run "python COMP3221_FLClient.py <Client-id> <Port-client> <Opt-method>". Client-id is an integer representing an id number, Port-client should start at 6001 and increment for each additional client and Opt-method is a flag to specify the local model (0 runs GD and 1 runs mini-batch GD).

Once the 30 seconds has begun, the global communication rounds 1 through to 100 will start.

During this process, clients will write the training loss and testing accuracy every round to a local client{id}_log.txt file, as well as output this data to the terminal. The server will also aggregate all client data and write this to an evaluation_log.txt file only. The user can inspect these files once the server and models have run and shutdown.
