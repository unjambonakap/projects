
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export ROOT=$(realpath $DIR/..)
export DB_FILE=$ROOT/data/main.db
export DB_URI=sqlite:///$DB_FILE
