import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
     
    <div>
        <header className="title">
          Lux AI Season 3 Model Simulator
        </header>

        {/* Main layout */}
        <div className="main-container">

          {/* Left panel: Environment Parameters */}
          <div className="left-panel">
            <div classname = "L-parameters">
            <h2>Environment Parameters</h2>

            <label htmlFor="asteroid-speed">Asteroid Speed</label>
            <input type="range" id="asteroid-speed" min="0" max="100" list = "markers"/>
            <datalist id = "A-markers"> 
              <option value = "0"></option>
              <option value = "25"></option>
              <option value = "50"></option>
              <option value = "75"></option>
              <option value = "100"></option>
            </datalist>
            <label htmlFor="asteroid-density">Asteroid Density</label>
            <input type="range" id="asteroid-density" min="0" max="100" list = "markers"/>
            <datalist id = "B-markers"> 
              <option value = "0"></option>
              <option value = "25"></option>
              <option value = "50"></option>
              <option value = "75"></option>
              <option value = "100"></option>
            </datalist>
            <label htmlFor="nebula-speed">Nebula Speed</label>
            <input type="range" id="nebula-speed" min="0" max="100" />
            
            <label htmlFor="nebula-density">Nebula Density</label>
            <input type="range" id="nebula-density" min="0" max="100" />

            <label htmlFor="relic-node-spawn">Relic Node Spawn Rate</label>
            <input type="range" id="relic-node-spawn" min="0" max="100" />

            <label htmlFor="energy-tile-density">Energy Tile Density</label>
            <input type="range" id="energy-tile-density" min="0" max="100" />

            <label htmlFor="energy-expenditure">Energy Expenditure</label>
            <input type="range" id="energy-expenditure" min="0" max="100" />
            </div>
          </div>
          <div classname = "simulation-space"> 
            <lable>simulation space</lable>
          </div>
          {/* Right panel: Game Parameters */}
          <div className="right-panel">
            <h2>Game Parameters</h2>

            <label htmlFor="adversarial-mode">Adversarial Mode:</label>
            <select id="adversarial-mode">
              <option value="model1">Model Selection 1</option>
              <option value="model2">Model Selection 2</option>
            </select>

            <label>Starting Points Distribution</label>
            <input type="number" min="0" max="100" defaultValue="1" />
            <input type="number" min="0" max="100" defaultValue="1" />

            <label>Total Matches</label>
            <input type="number" min="1" max="100" defaultValue="1" />
          </div>
        </div>

        {/* Footer with start button */}
        <footer>
          <button className="start-btn">Start Simulation</button>
        </footer>
    </div>
  );
}

export default App
