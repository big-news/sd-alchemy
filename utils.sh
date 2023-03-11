export C_RESET='\033[0m'
export C_RED='\033[0;31m'
export C_GREEN='\033[0;32m'
export C_BLUE='\033[0;34m'
export C_YELLOW='\033[1;33m'

function println() {
  echo -e "$1"
}

function errorln() {
  println "${C_RED}${1}${C_RESET}"
}

function successln() {
  println "${C_GREEN}${1}${C_RESET}"
}

function infoln() {
  println "${C_BLUE}${1}${C_RESET}"
}

function warnln() {
  println "${C_YELLOW}${1}${C_RESET}"
}

function fatalln() {
  errorln "$1"
  exit 1
}

export -f println
export -f errorln
export -f successln
export -f infoln
export -f warnln
export -f fatalln