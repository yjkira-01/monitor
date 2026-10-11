import { Component, type ReactNode } from 'react'

export default class SpatialBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed ? <div className="evidence-empty" role="alert">三维视图暂不可用，视频与识别继续运行。<button onClick={() => this.setState({ failed: false })}>重新加载三维视图</button></div> : this.props.children
  }
}
