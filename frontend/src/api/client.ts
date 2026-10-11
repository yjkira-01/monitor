const API_BASE = (import.meta.env.VITE_API_BASE as string) || '/api'

import type { AlertData } from '../types'

export function getToken() {
  return localStorage.getItem('access_token')
}

export function setToken(token: string) {
  localStorage.setItem('access_token', token)
}

export async function login(username: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) throw new Error('登录失败')
  return res.json() as Promise<{ access_token: string; token_type: string; role: string }>
}

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  const token = getToken()
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token || ''}`)
  const response = await fetch(input, { ...init, headers })
  if (response.status === 401 && getToken() === token) {
    localStorage.removeItem('access_token')
    window.dispatchEvent(new Event('auth-expired'))
  }
  return response
}

export async function getAlerts(limit = 50) {
  const res = await authFetch(`${API_BASE}/alerts?limit=${limit}`)
  if (!res.ok) throw new Error('获取预警列表失败')
  return res.json() as Promise<AlertData[]>
}

export async function logout() {
  localStorage.removeItem('access_token')
}
