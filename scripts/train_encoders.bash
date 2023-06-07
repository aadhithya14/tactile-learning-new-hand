#!/bin/bash
# sessname="temporal_joint_trainings"
sessname=$1

# cd ~/data
# mkdir "$1"
cd ../

# Create a new session named "$sessname", and run command
tmux new-session -d -s "$sessname" || echo "going to the session $sessname"
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=bowl_picking learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=plier_picking learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=card_flipping learner.total_loss_type=$2 device=$3" Enter
# tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=$2 device=$3" Enter

tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=card_turning learner.total_loss_type=contrastive device=$2" Enter
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=joint device=$2" Enter
tmux send-keys -t "$sessname" "python train_nondist.py train_epochs=200 object=peg_insertion learner.total_loss_type=contrastive device=$2" Enter

# TODO: Bowl picking gave error since you didn't change the view - fix that

# Attach to session named "$sessname"
#tmux attach -t "$sessname"