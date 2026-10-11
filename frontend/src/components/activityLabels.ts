export const activityNames: Record<string, string> = {
  walk: '行走', fall: '跌倒过程', fallen: '跌倒后倒地', sit_down: '正在坐下', sitting: '坐姿',
  lie_down: '主动躺下', lying: '躺姿', stand_up: '正在起身', standing: '站姿', other: '其他活动', adl: '正常活动',
}
export type Activity = { label: string; confidence: number; scores: Record<string, number>; basis: string }
