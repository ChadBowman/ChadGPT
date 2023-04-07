import React, { useState, useEffect } from "react";
import {
    Button,
    Box,
    Select,
    VStack,
} from "@chakra-ui/react";

export const EvalContext = React.createContext({
    fetchModels: () => {}
})

export default function Evals() {
    const [evals, setEvals] = useState([])
    const [targetModel, setTargetModel] = useState("")

    const [models, setModels] = useState([])

    const fetchModels = async () => {
        const response = await fetch("http://localhost:8000/lm")
        const m = await response.json()
        setModels(m)
        return m
    }

    const fetchEvals = async () => {
        if (!targetModel) {
            console.error("No model available")
            return
        }
        const response = await fetch(`http://localhost:8000/lm/${targetModel}/eval?tokens=1000?split_newlines=true`)
        const evals = await response.json()
        setEvals(evals.paragraphs)
    }

    useEffect(() => {
        const inner = async () => {
            const models = await fetchModels()
            setTargetModel(models[0])
        }
        inner()
    }, [])

    return (
      <EvalContext.Provider value={{fetchModels}}>
        <VStack spacing={3}>
            <label>
                Model
            </label>
            <Select placeholder="Select model"
                onChange={(event) => {setTargetModel(event.target.value)}}>
                {models && models.map((model, index) => (
                    <option value={model} key={index}>{model}</option>
                ))}
            </Select>
            <Box fontSize="md" overflowY="auto" minWidth="700px" height="500px" borderWidth="1px" borderRadius="md" px={4}>
                {evals && evals.map((paragraph, index) => (
                    <React.Fragment key={index}>
                    {paragraph ? <span><br />{paragraph}</span> : <br />}
                    </React.Fragment>
                ))}
            </Box>
            <Button onClick={fetchEvals}>Generate</Button>
        </VStack >
      </EvalContext.Provider>
    )
}
