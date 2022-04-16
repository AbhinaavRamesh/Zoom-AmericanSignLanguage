import React, { useState } from 'react'
import Landing from './screens/Landing';

function ZoomASL(){
    const [pageState,setPageState]=useState(1);
    function goToNextPage(){
        setPageState(2)
    }
    if(pageState === 1){
        return <Landing goToNextPage={goToNextPage}/>
    }
}

export default ZoomASL;