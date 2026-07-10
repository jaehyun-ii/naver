import { Routes, Route } from "react-router-dom";
import { Shell } from "./components/Shell";
import { ALL_ITEMS } from "./nav";
import Home from "./pages/Home";
import Playground from "./pages/Playground";
import Datasets from "./pages/Datasets";
import Documents from "./pages/Documents";
import Dashboard from "./pages/Dashboard";
import Releases from "./pages/Releases";
import Wizard from "./pages/Wizard";
import Pipeline from "./pages/Pipeline";
import Hpo from "./pages/Hpo";
import Prompts from "./pages/Prompts";
import Rag from "./pages/Rag";
import Review from "./pages/Review";
import Safety from "./pages/Safety";
import EvalVersions from "./pages/EvalVersions";
import Benchmark from "./pages/Benchmark";
import Drift from "./pages/Drift";
import Serving from "./pages/Serving";
import Tracking from "./pages/Tracking";
import Storage from "./pages/Storage";
import Traces from "./pages/Traces";
import Keys from "./pages/Keys";
import Security from "./pages/Security";
import Placeholder from "./pages/Placeholder";

// 구현된 화면의 경로 → 컴포넌트 매핑. 나머지는 Placeholder로 렌더한다.
const IMPLEMENTED: Record<string, React.ComponentType> = {
  "/": Home,
  "/play": Playground,
  "/data": Datasets,
  "/documents": Documents,
  "/dash": Dashboard,
  "/rel": Releases,
  "/wizard": Wizard,
  "/pipe": Pipeline,
  "/hpo": Hpo,
  "/prompt": Prompts,
  "/rag": Rag,
  "/review": Review,
  "/safety": Safety,
  "/evalv": EvalVersions,
  "/bench": Benchmark,
  "/drift": Drift,
  "/serving": Serving,
  "/mlflow": Tracking,
  "/storage": Storage,
  "/traces": Traces,
  "/keys": Keys,
  "/sec": Security,
};

export default function App() {
  return (
    <Shell>
      <Routes>
        {ALL_ITEMS.map((item) => {
          const Comp = IMPLEMENTED[item.path];
          return (
            <Route
              key={item.path}
              path={item.path}
              element={Comp ? <Comp /> : <Placeholder title={item.label} />}
            />
          );
        })}
        <Route path="*" element={<Placeholder title="페이지를 찾을 수 없음" />} />
      </Routes>
    </Shell>
  );
}
