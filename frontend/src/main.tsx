import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { events, installPlayer } from "./events";
import "./styles.css";

events.start();
installPlayer();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
