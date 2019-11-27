console.log('main.js loaded')

async function getData () {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  const carsData = await carsDataReq.json()
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower
  }))
    .filter(car => car.mpg != null && car.horsepower != null)

  return cleaned
}

async function run () {
  const data = await getData()
  console.log('data', data)
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }))

  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'horsepower',
      yLabel: 'mpg',
      height: 300
    }
  )

  const model = createModel()
  tfvis.show.modelSummary({ name: 'Model Summary', tab: 'Model Inspection' }, model)

  const tensorData = convertToTensor(data)
  const { inputs, labels } = tensorData
  await trainModel(model, inputs, labels)
  console.log('Done Training')
  // console.log(convertToTensor(values))
}

function createModel () {
  const model = tf.sequential()

  // Defining HIDDEN LAYER
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))

  // Defining output layer
  model.add(tf.layers.dense({ units: 1 }))

  return model
}

function convertToTensor (data) {
  return tf.tidy(() => {
    // 1) Shuffle data
    tf.util.shuffle(data)

    // 2) Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])
    console.log('inputTensor', inputTensor)

    // 3) Normailize the data
    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  })
}

async function trainModel (model, inputs, labels) {
  // compile
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  const batchSize = 32
  const epochs = 50

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}

function testModel (mode, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = model.predict(xs.reshape([100, 1]))

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin)

    const unNormPreds = performance
      .mul(labelMax.su(labelMin))
      .add(labelMin)

    return [unNormXs.dataSync(), unNormPres, dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, r) => {
    return { x: val, y: preds[i] }
  })

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg
  }))

  tfvis.rednder.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}

document.addEventListener('DOMContentLoaded', run)
