import commonApi from './commonApi'

export function listAccessKeys() {
    return commonApi.get('/access-keys')
}

export function createAccessKey(name) {
    return commonApi.post('/access-keys', {name})
}

export function revokeAccessKey(accessKeyId) {
    return commonApi.delete(`/access-keys/${accessKeyId}`)
}
