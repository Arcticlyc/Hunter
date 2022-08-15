import numpy as np
from PIL import Image
import hashlib
from numba import jit
import time


# 随机打乱
def amess(arrlength, ast):
    arr = np.linspace(0, arrlength-1, arrlength, dtype=int)
    for i in range(arrlength-1, 0, -1):
        content = (ast + str(i)).encode()
        md5hash = hashlib.md5(content)
        md5 = (md5hash.hexdigest())[:7].upper()
        rand = int(md5, 16) % (i + 1)
        temp = arr[rand]
        arr[rand] = arr[i]
        arr[i] = temp
    return arr


@jit(nopython=True)
def get_img_1(img_li, sx, sy, xl, yl):
    hit, wid, z = img_li.shape

    ssx = wid / 32
    ssy = hit / 32

    new_img = np.zeros((hit, wid, 4))
    for i in range(wid):
        for j in range(hit):
            m, n = i, j
            m = (xl[(int(n / ssy)) % sx] * ssx + m) % wid
            # print(m)
            m = xl[int(m / ssx)] * ssx + m % ssx
            n = (yl[(int(m / ssx)) % sy] * ssy + n) % hit
            n = yl[int(n / ssy)] * ssy + n % ssy
            m, n =int(m), int(n)

            new_img[(m + n * wid) // wid][(m + n * wid) % wid] = img_li[(i + j * wid) // wid][(i + j * wid) % wid]

    return new_img


@jit(nopython=True)
def get_img_2(img_li, xl):
    hit, wid, z = img_li.shape
    new_img = np.zeros((hit, wid, 4))
    for i in range(wid):
        for j in range(hit):
            m, n = i, j
            m = (xl[n % wid] + m) % wid
            m = xl[m]

            new_img[(m + n * wid) // wid][(m + n * wid) % wid] = img_li[(i + j * wid) // wid][(i + j * wid) % wid]

    return new_img


@jit(nopython=True)
def get_img_3(img_li, xl, yl):
    hit, wid, z = img_li.shape
    new_img = np.zeros((hit, wid, 4))
    for i in range(wid):
        for j in range(hit):
            m, n = i, j
            m = (xl[n % wid] + m) % wid
            m = xl[m]
            n = (yl[m % hit] + n) % hit
            n = yl[n]

            new_img[(m + n * wid) // wid][(m + n * wid) % wid] = img_li[(i + j * wid) // wid][(i + j * wid) % wid]

    return new_img


@jit(nopython=True)
def get_img_4(img_li, arrayaddress):
    hit, wid, z = img_li.shape
    new_img = np.zeros((hit, wid, 4))
    for i in range(wid):
        for j in range(hit):
            m = arrayaddress[i]
            new_img[(m + j * wid) // wid][(m + j * wid) % wid] = img_li[(i + j * wid) // wid][(i + j * wid) % wid]
    return new_img


def get_img_5(img_li, key):
    hit, wid, z = img_li.shape
    new_img_1 = np.zeros((hit, wid, 4))
    new_img_2 = np.zeros((hit, wid, 4))

    x = key
    for i in range(wid):
        arrayaddress_hit = produceLogistic(x, hit)
        x = arrayaddress_hit[hit - 1][0]
        arrayaddress_hit.sort(key=lambda p: p[0])
        arrayaddress_hit = np.array([arr[1] for arr in arrayaddress_hit])
        for j in range(hit):
            n = arrayaddress_hit[j]
            new_img_1[(i + n * wid) // wid][(i + n * wid) % wid] = img_li[(i + j * wid) // wid][(i + j * wid) % wid]

    x = key
    for j in range(hit):
        arrayaddress_wid = produceLogistic(x, wid)
        x = arrayaddress_wid[wid - 1][0]
        arrayaddress_wid.sort(key=lambda p: p[0])
        arrayaddress_wid = np.array([arr[1] for arr in arrayaddress_wid])
        for i in range(wid):
            m = arrayaddress_wid[i]
            new_img_2[(m + j * wid) // wid][(m + j * wid) % wid] = new_img_1[(i + j * wid) // wid][(i + j * wid) % wid]

    return new_img_2


# 1. 方块混淆
def decryptB2(img_li, key):
    sx, sy = 32, 32
    xl = amess(sx, key)
    yl = amess(sy, key)

    new_img = get_img_1(img_li, sx, sy, xl, yl)
    return new_img


# 2. 行像素混淆
def decryptC2(img_li, key):
    wid, hit, z = img_li.shape

    xl = amess(wid, key)

    new_img = get_img_2(img_li, xl)
    return new_img


# 3. 像素混淆
def decryptC(img_li, key):
    wid, hit, z = img_li.shape

    xl = amess(wid, key)
    yl = amess(hit, key)

    new_img = get_img_3(img_li, xl, yl)
    return new_img


# PicEncrypt算法
def produceLogistic(key, wid):
    x = key
    l = [[x, 0]]
    for i in range(1, wid):
        x = 3.9999999 * x * (1 - x)
        l.append([x, i])
    return l


# 4. 兼容PicEncrypt: 行模式
def decryptPE1(img_li, key):
    hit, wid, z = img_li.shape

    arrayaddress = produceLogistic(key, wid)
    arrayaddress.sort(key=lambda p: p[0])
    arrayaddress = np.array([arr[1] for arr in arrayaddress])

    new_img = get_img_4(img_li, arrayaddress)
    return new_img


# 5. 兼容PicEncrypt: 行+列模式
def decryptPE2(img_li, key):
    new_img = get_img_5(img_li, key)
    return new_img


# 主函数
def main(mode, path, key, out):
    '''解混淆主函数
    1. 方块混淆
    2. 行像素混淆
    3. 像素混淆
    4. 兼容PicEncrypt: 行模式
    5. 兼容PicEncrypt: 行+列模式'''
    img = Image.open(path)

    if img.mode != 'RGBA':
            img = img.convert('RGBA')
    img_li = np.array(img)

    if mode == '1':
        new_img = decryptB2(img_li, key)

    elif mode == '2':
        new_img = decryptC2(img_li, key)

    elif mode == '3':
        new_img = decryptC(img_li, key)

    elif mode == '4':
        key = float(key)
        new_img = decryptPE1(img_li, key)

    elif mode == '5':
        key = float(key)
        new_img = decryptPE2(img_li, key)

    img = Image.fromarray(np.uint8(new_img))
    img.save(out)
    print(f'文件{out}已存入。')


if __name__ == '__main__':
    image = input('Path：')
    pword = input('Password(0-1)：')
    save_path = input('Save to(不含后缀)：') + '.png'
    mode = input('1. 方块混淆\n2. 行像素混淆\n3. 像素混淆\n4. 兼容PicEncrypt: 行模式\n5. 兼容PicEncrypt: 行+列模式\n输入解混淆模式：')
    start_time = time.time()

    main(mode, image, pword, save_path)
    input('\nDecryption Finished...\nTime: {:.6f} second(s)\nPress any key...'.format(
        time.time()-start_time))
