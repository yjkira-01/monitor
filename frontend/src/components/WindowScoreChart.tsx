import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { WindowResult } from './VideoWorkbench'

export default function WindowScoreChart({ history }: { history: WindowResult[] }) {
  return <ResponsiveContainer width="100%" height={180}><LineChart data={history} margin={{ top: 12, right: 12, bottom: 0, left: -20 }}>
    <XAxis dataKey="window_end_s" tickFormatter={value => `${value}s`} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={25} />
    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
    <Tooltip labelFormatter={value => `窗口结束 ${value}s`} formatter={value => [Number(value).toFixed(3), '跌倒特征分数']} />
    <ReferenceLine y={0.5} stroke="#b98752" strokeDasharray="4 4" />
    <Line type="linear" dataKey="fall_score" stroke="#35745f" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
  </LineChart></ResponsiveContainer>
}
