import '@mantine/core/styles.css';

import {MantineProvider} from '@mantine/core';
import {Router} from './Router';
import {theme} from './theme';
import StateContainer from './components/StateContainer';
import LocalStateContainer from './components/LocalStateContainer';

export default function App() {
  return (
    <LocalStateContainer>
      <StateContainer>
        <MantineProvider theme={theme}>
          <Router />
        </MantineProvider>
      </StateContainer>
    </LocalStateContainer>
  );
}
