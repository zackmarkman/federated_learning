# COMP3221 Federated Learning Assignment 2
# This README file details how to run the server and K clients

NOTE: we have run and tested these programs on both Linux and Mac environments. Please use one of these and NOT Windows to ensure consistency and avoid unforeseen errors.

Run the server with "python COMP3221_FLServer.py <Port-server> <Sub-client>".
 - Port-server should be set to 6000
 - Sub-client flag should be set to 0 or 1
   - 0 means the server will aggregate all client models
   - 1 means the server will aggregate 2 client models at random

Run a client with "python COMP3221_FLClient.py <Client-id> <Port-client> <Opt-method>"
 - Client-id is the client number
 - Port-client should start at 6001 and increment for each additional client
 - Opt-method should be set to 0 or 1
   - 0 runs the local model as gradient descent
   - 1 runs the local mode as mini-batch gradient descent

The server will then wait 30 seconds for more clients to initialise and connect before starting FedAvg and the 100 communication rounds.

In each communication round, clients will write their training loss and testing accuracy a local client<id>_log.txt file.

The server will also aggregate client model data and write this to an evaluation_log.txt file.

If a client disconnects, it will not be included in the following round. However, clients can reconnect to the server and will join the next communication round.
