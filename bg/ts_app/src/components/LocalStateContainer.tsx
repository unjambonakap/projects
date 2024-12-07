import React, {useState, Reducer} from 'react';
import {StateConsumer} from '../components/StateContainer';

interface IState {
  actionCount: number;
}

interface IContext {
  state: IState;
  actions: {
    notifyAction: () => void;
  };
}

export const INITIAL_STATE: IState = {actionCount: 0};

const LocalStateContext = React.createContext<IContext>({
  state: INITIAL_STATE,
  actions: {notifyAction: () => {}},
});

const localStorageKey = 'state';

function getInitialState() {
  const state = localStorage.getItem(localStorageKey);
  return state ? JSON.parse(state) : INITIAL_STATE;
}

class LocalStateContainer extends React.PureComponent<{}, IState> {
  state = getInitialState();

  useEffect() {}

  render() {
    const context = {
      state: this.state,
      actions: {
        notifyAction: () => {
          console.log(`Register action ${this.state.actionCount}`);

          var nstate = {...this.state, actionCount: this.state.actionCount + 1};
          this.setState(nstate);
          localStorage.setItem(localStorageKey, JSON.stringify(nstate));
        },
      },
    };
    return (
      <LocalStateContext.Provider value={context}>
        {this.props.children}
      </LocalStateContext.Provider>
    );
  }
}

export default LocalStateContainer;
export const LocalStateConsumer = LocalStateContext.Consumer;

export const AllStateConsumer = ({children}) => (
  <>
    <StateConsumer>
      {stateConsumer => (
        <LocalStateConsumer>
          {localStateConsumer =>
            children({
              ctx: stateConsumer,
              localCtx: localStateConsumer
            })
          }
        </LocalStateConsumer>
      )}
    </StateConsumer>
  </>
);
