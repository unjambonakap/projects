import {ColorSchemeToggle} from '../components/ColorSchemeToggle/ColorSchemeToggle';
import {Welcome} from '../components/Welcome/Welcome';
import {Button} from '@mantine/core';
import {AllStateConsumer} from '../components/LocalStateContainer';
import {StateConsumer} from '../components/StateContainer';
import * as api from '../utils/api';
import React, {useState, useEffect} from 'react';


export const GameSelectMenu = (ctx) => (
        <>
          <p> Create a game</p>
          <Button color="#841584" 
          onClick={() => ctx.actions.createGame()}/>
        </>
  );

export const GameScreen = (ctx) =>  (
    <>
      <p> game screen </p>
      <iframe src={ctx.actions.getMapDisplayURL()}
      width="100%"
      height="80%"/>
    </>

);
   

export function HomePage() {
  return (
    <>
      <AllStateConsumer>
        {({ctx, localCtx}) => (
          <>
            <p>
            Status of state= {JSON.stringify(ctx.state)} <br/>
             Status of local state= {JSON.stringify(localCtx.state)}
            </p>
            <Button
              onClick={() => {
                ctx.actions.setActiveGame('123');
                localCtx.actions.notifyAction();
              }}
              color="#841584"
            />
            {ctx.state.game != undefined ? GameScreen(ctx) : GameSelectMenu(ctx) }
          </>
        )}
      </AllStateConsumer>
    </>
  );
}
