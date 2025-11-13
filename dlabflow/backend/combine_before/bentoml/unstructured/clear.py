../../port-kill.sh
lsof -ti :30000 | xargs kill -9
