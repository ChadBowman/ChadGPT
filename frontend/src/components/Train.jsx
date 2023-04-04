import React, { useState } from "react";
import { EvalContext } from "./Eval";
import {
    Button,
    Input,
    VStack,
    NumberInput,
    NumberInputField,
} from "@chakra-ui/react";


function HyperParameterInput({
    label,
    min,
    max,
    onChange,
    defaultValue
}) {
    const toNumber = (fn) => {
        return function(arg) {
            return fn(Number(arg));
        }
    }
    return (
        <NumberInput min={min} max={max} onChange={toNumber(onChange)} defaultValue={defaultValue} >
            {label}
            <NumberInputField />
        </NumberInput>
    )
}

export default function Train() {
    const { fetchModels } = React.useContext(EvalContext)
    const defaultDataset = "shakespeare.txt"
    const defaultTokenizer = "character"
    const defaultName = "test"
    const defaultNLayer = 6
    const defaultNEmbed = 512
    const defaultNHeads = 8
    const defaultBlockSize = 64
    const defaultDropout = 0.1
    const defaultLR = 0.0001
    const defaultMaxIter = 3000
    const defaultBatchSize = 6

    const [dataset, setDataset] = useState(defaultDataset)
    const [tokenizer, setTokenizer] = useState(defaultTokenizer)
    const [name, setName] = useState(defaultName)
    const [nLayer, setNLayer] = useState(defaultNLayer)
    const [nEmbed, setNEmbed] = useState(defaultNEmbed)
    const [nHeads, setNHeads] = useState(defaultNHeads)
    const [blockSize, setBlockSize] = useState(defaultBlockSize)
    const [batchSize, setBatchSize] = useState(defaultBatchSize)
    const [dropout, setDropout] = useState(defaultDropout)
    const [learningRate, setLearningRate] = useState(defaultLR)
    const [maxIters, setMaxIters] = useState(defaultMaxIter)

    const handleName = (event) =>  {
        setName(event.target.value)
    }

    const handleSubmit = (event) => {
        event.preventDefault()
        const trainingParams = {
            "dataset": dataset,
            "tokenizer": tokenizer,
            "hyperparameters": {
                "vocab_size": 65,
                "block_size": blockSize,
                "n_heads": nHeads,
                "n_embed": nEmbed,
                "n_layer": nLayer,
                "dropout": dropout,
                "batch_size": batchSize,
                "max_iters": maxIters,
                "learning_rate": learningRate,
                "eval_interval": 500,
                "eval_iters": 200
            }
        }
        fetch(`http://localhost:8000/lm/${name}/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(trainingParams)
        }).then(fetchModels)
    }

    return (
        <form onSubmit={handleSubmit}>
            <VStack spacing={1} align="left">
                <label>
                    Model name
                    <Input type="text"  defaultValue="" onChange={handleName} required={true}/>
                </label>
                <HyperParameterInput label="Number of attention layers"
                    min={1} max={12} defaultValue={defaultNLayer}
                    onChange={(param) => setNLayer(param)} />
                <HyperParameterInput label="Embedding size"
                    min={8} max={1024} defaultValue={defaultNEmbed}
                    onChange={(param) => setNEmbed(param)} />
                <HyperParameterInput label="Number of heads"
                    min={1} max={32} defaultValue={defaultNHeads}
                    onChange={(param) => setNHeads(param)} />
                <HyperParameterInput label="Block size"
                    min={8} max={1028} defaultValue={defaultBlockSize}
                    onChange={(param) => setBlockSize(param)} />
                <HyperParameterInput label="Dropout"
                    min={0.0} max={0.4} defaultValue={defaultDropout}
                    onChange={(param) => setDropout(param)} />
                <HyperParameterInput label="Learning rate"
                    min={0} max={0.001} defaultValue={defaultLR}
                    onChange={(param) => setLearningRate(param)} />
                <HyperParameterInput label="Training iterations"
                    min={100} max={300000} defaultValue={defaultMaxIter}
                    onChange={(param) => setMaxIters(param)} />
                <HyperParameterInput label="Batch size"
                    min={1} max={32} defaultValue={defaultBatchSize}
                    onChange={(param) => setBatchSize(param)} />
            </VStack>
            <Button type="submit">Train Model</Button>
        </form>
    )
}
