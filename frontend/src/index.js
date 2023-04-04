import React from "react";
import { render } from 'react-dom';
import { 
    ChakraProvider,
    HStack,
} from "@chakra-ui/react";

import Header from "./components/Header";
import Eval from "./components/Eval";
import Train from "./components/Train";

function App() {
    return (
        <ChakraProvider>
            <Header />
            <HStack spacing="20px" ml={10}>
                <Eval />
                {/* <Train /> hide for now */}
            </HStack>
        </ChakraProvider>
    )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)
