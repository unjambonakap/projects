import React, {useState, Reducer} from 'react';
import * as api from '../utils/api';
import parse from 'html-react-parser';

interface Player { 
  id: string
}

interface Game { 
  id: string
  players: Player[]

}
interface IState {
  game: Game | undefined
  curPlayer: Player | undefined
  
}

interface IContext {
  state: IState;
  actions: {
    setActiveGame: (gameId: string) => void;
    createGame: () => void;
    getMapDisplayURL: () => string ;
  };
}

export const INITIAL_STATE: IState ={game:undefined, curPlayer: undefined};

const StateContext = React.createContext<IContext>({
  state: INITIAL_STATE,
  actions: {
    setActiveGame: _ => {},
    createGame: () => {},
    getMapDisplayURL: () => "",
  },
});

class StateContainer extends React.PureComponent<{}, IState> {
  state = INITIAL_STATE;

  render() {
    const context = {
      state: this.state,
      actions: {
        setActiveGame: (gameId: string) => {
          console.log(`Set active game ${gameId}`);
          api.get(`/user/setActiveGame/${gameId}`).then(res => {
            console.log(`DONE ${res}`);
          });

        },
        createGame: () => {
          api.get(`/game/create`).then(game => {
            this.setState({...this.state, game:game, curPlayer:game.players[-1]});
          });
        },

        getMapDisplayURL: () => api.getURL(`/game/op/${this.state.game.id}/map_display`)
      },
    };
    return (
      <StateContext.Provider value={context}>
        {this.props.children}
      </StateContext.Provider>
    );
  }
}

export default StateContainer;
export const StateConsumer = StateContext.Consumer;
