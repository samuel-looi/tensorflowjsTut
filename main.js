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
getData()
  .then(data => console.log(data))

async function run () {
  const data = await getData()
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
}

function createModel () {
  const model = tf.sequential()

  // Defining input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))

  // Defining output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }))

  return model
}

document.addEventListener('DOMContentLoaded', run)
