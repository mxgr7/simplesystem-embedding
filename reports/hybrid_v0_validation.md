# hybrid_v0 — validation

- search-api: `http://localhost:8001`
- dense collection: `offers`
- timestamp: 2026-04-27 11:59:37 UTC

## 1. Classifier precision (top-200 PostHog queries)

- total queries inspected: 200
- queries flagged as strict identifier: 1
- share of flagged volume: 0.1%

Flagged queries (all should be real identifiers — eyeball):

| query | events |
|---|---|
| 4003773035466 | 93 |

## 2. Codes hits on EAN-shaped queries (BM25 mode, k=5)

- queries probed: 100
- with at least one hit: 86 / 100 (86.0%)

| query | events | hits | top score |
|---|---|---|---|
| 4003773035466 | 93 | 5 | 16.893 |
| 15010903 | 30 | 0 | — |
| 00000000 | 22 | 0 | — |
| 4014651502635 | 17 | 5 | 17.219 |
| 10585834 | 17 | 3 | 18.796 |
| 72440355 | 17 | 0 | — |
| 4008496559312 | 14 | 5 | 17.428 |
| 4004764007967 | 12 | 5 | 16.809 |
| 58211032 | 12 | 5 | 16.326 |
| 72030180 | 11 | 1 | 19.700 |
| 5708997356906 | 11 | 5 | 17.733 |
| 4977766685177 | 10 | 5 | 17.383 |
| 4050821808442 | 10 | 5 | 15.185 |
| 58211041 | 9 | 5 | 16.326 |
| 5411313450133 | 9 | 5 | 16.730 |
| 4007126023117 | 9 | 5 | 14.962 |
| 4014651367715 | 9 | 5 | 16.109 |
| 4008190072506 | 9 | 5 | 16.987 |
| 5099206112094 | 9 | 5 | 16.159 |
| 4050821808435 | 9 | 5 | 15.206 |
| 4011376750570 | 9 | 5 | 17.439 |
| 270000231902 | 9 | 1 | 19.700 |
| 270000131602 | 9 | 1 | 19.700 |
| 72400078 | 8 | 0 | — |
| 4046356329781 | 8 | 5 | 15.364 |
| 4011160464034 | 8 | 5 | 20.187 |
| 4031026106250 | 8 | 5 | 15.535 |
| 39269097 | 8 | 1 | 19.700 |
| 270000131603 | 8 | 1 | 19.700 |
| 4068400003324 | 8 | 5 | 16.523 |

## 3. Free-text regression (hybrid vs vector top-24)

- queries: 100
- rejected by classifier: 100 / 100
- median dense∩hybrid overlap: 100.0%
- mean   dense∩hybrid overlap: 99.9%

| query | overlap | dense_n | hybrid_n | codes_added |
|---|---|---|---|---|
| None | 100% | 24 | 24 | 24 |
| Halter Bügelmessschraube | 100% | 24 | 24 | 24 |
| S | 100% | 24 | 24 | 24 |
| s | 100% | 24 | 24 | 24 |
| 0 | 100% | 24 | 24 | 24 |
| 1 | 100% | 24 | 24 | 24 |
| A | 100% | 24 | 24 | 24 |
| 6 | 100% | 24 | 24 | 24 |
| M | 100% | 24 | 24 | 24 |
| 2 | 100% | 24 | 24 | 24 |
| 3 | 100% | 24 | 24 | 24 |
| 7 | 100% | 24 | 24 | 24 |
| m | 100% | 24 | 24 | 24 |
| 4 | 100% | 24 | 24 | 24 |
| B | 100% | 24 | 24 | 24 |
| b | 100% | 24 | 24 | 24 |
| a | 100% | 24 | 24 | 24 |
| T | 100% | 24 | 24 | 24 |
| sch | 100% | 24 | 24 | 24 |
| P | 100% | 24 | 24 | 24 |

## 4. 0-result fallback probes

| query | hits | path | fallback? | classifier strict? |
|---|---|---|---|---|
| din912 | 24 | fallback | yes | yes |
| 9999999999998 | 3 | strict | no | yes |
| rj45zzz | 24 | fallback | yes | yes |

## 5. Top-of-fused-page eyeball (hybrid_classified)

- queries shown: 25

### 0
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 2563cb94017db15d3989612a2be9d6ca | 0.0164 | rrf |
| 2 | 5ecbd6fa8f48578e36d02cb54cbe5899 | 0.0161 | rrf |
| 3 | 016f93f7b35b2e258b5cdb8d2ccb2dc6 | 0.0159 | rrf |
| 4 | 235c4f5b97979f1d3683b6fe9dd35457 | 0.0156 | rrf |
| 5 | 0a37773f33a4645362dd600d6ac3e93a | 0.0154 | rrf |
| 6 | 716ad7f0bca7bd6bfb5805195cf0eb05 | 0.0152 | rrf |
| 7 | a2bbaebcc58769c27677bbb07ae683a6 | 0.0149 | rrf |
| 8 | 0a4730935e40ce914534eaa1c51606b9 | 0.0147 | rrf |
| 9 | 372fd66b17adfa9ee63f724784dc731a | 0.0145 | rrf |
| 10 | e24916709f803b910d6b71e74735bc79 | 0.0143 | rrf |

### 1
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 2d2c32a26a08df95348127384639a6fb | 0.0164 | rrf |
| 2 | 63e0d7f38640f4389e640a4ae049deee | 0.0161 | rrf |
| 3 | 8b7c919131e5df32ca95737ebe062e6c | 0.0159 | rrf |
| 4 | 4e6209ce5fcf575c2e53e5c6d0211869 | 0.0156 | rrf |
| 5 | c09844e7d2be0b9debeb819e06f46aee | 0.0154 | rrf |
| 6 | c670061fb71c88e13c6a64d0c6bc29b5 | 0.0152 | rrf |
| 7 | 9444d0aba523ab1a471caad00f8aa696 | 0.0149 | rrf |
| 8 | e32bdbc02d3cad2a990ba57eea1be510 | 0.0147 | rrf |
| 9 | 9d36703096465eec9fe5a70fb8fbb6dd | 0.0145 | rrf |
| 10 | 7f87e16afced124afa66d6a89d9f8a25 | 0.0143 | rrf |

### 6
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 607c645355970bbc669ac057fd0508c6 | 0.0164 | rrf |
| 2 | 254f477a9f054febde102d81765f6319 | 0.0161 | rrf |
| 3 | a91f2285a19f3e0b5c510394007bb41c | 0.0159 | rrf |
| 4 | 5663f4830be0682b45370719180d5f95 | 0.0156 | rrf |
| 5 | 247c07613bab47c42596474e7142539f | 0.0154 | rrf |
| 6 | dc17e96f9ed93005b6fd2edd85b88209 | 0.0152 | rrf |
| 7 | 76459549837174503acaaa89730c8faa | 0.0149 | rrf |
| 8 | 932041c210ba4826c852bf0b307ecff8 | 0.0147 | rrf |
| 9 | ae97f2c5409f5e4f96155da5852d8180 | 0.0145 | rrf |
| 10 | 3319c7ca192c72ec44772fba1061ac97 | 0.0143 | rrf |

### 2
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 8270c12de90ed92e3cc0b4ceb1868ab6 | 0.0164 | rrf |
| 2 | da071ef4aced8a9b1df58ab70b36b9ab | 0.0161 | rrf |
| 3 | 49c7f9bf01de2f52cbdfa65eb3d57271 | 0.0159 | rrf |
| 4 | 75870ca345cdc67d795524cc2322010a | 0.0156 | rrf |
| 5 | a511c720985d7b7f2b54caec8f3cac13 | 0.0154 | rrf |
| 6 | 12e87bf54496df1e30627c060405cfb6 | 0.0152 | rrf |
| 7 | 2d80c88b099b472052c7397c82c1dcb0 | 0.0149 | rrf |
| 8 | a6dd22b1dae306e6f63dd34ec09b1f81 | 0.0147 | rrf |
| 9 | 85674063f676361bf9b7581468fb7e41 | 0.0145 | rrf |
| 10 | b7b2e22983f5134f0b088852a09e13b2 | 0.0143 | rrf |

### 3
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 3811590be521b2e6f1557509f352cc20 | 0.0164 | rrf |
| 2 | 827387e05a95451c64d95918cabd7157 | 0.0161 | rrf |
| 3 | 0b80cd97555a1c6e887443126e2e6b06 | 0.0159 | rrf |
| 4 | fd1e4ded023d2085a83a1833f1e398ff | 0.0156 | rrf |
| 5 | d00a895b66dc3c83661b917ce9372ff5 | 0.0154 | rrf |
| 6 | 3524f62839f6646b88e6d74e581432e4 | 0.0152 | rrf |
| 7 | 573842a3d2cf1b8018cb81158bb9d334 | 0.0149 | rrf |
| 8 | d7c12b9c2c6df961f596d175af19e40e | 0.0147 | rrf |
| 9 | 119c842517293048d95ff71f314fd4fb | 0.0145 | rrf |
| 10 | bcb1b5a49156bcad3865ffac33ca8078 | 0.0143 | rrf |

### 7
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 78bbe16323cd37f0509b254407f8be24 | 0.0164 | rrf |
| 2 | fe82247756e254308253ae2c5b7da054 | 0.0161 | rrf |
| 3 | 5806685f398869a3d7ccf714d682f268 | 0.0159 | rrf |
| 4 | 50f247d87652b8ce506ddb7b47b5b824 | 0.0156 | rrf |
| 5 | 4f4ce8626a7892196b838628127ebf9d | 0.0154 | rrf |
| 6 | 6402e6b836e85d361a8f49debcbabcff | 0.0152 | rrf |
| 7 | 1d6c74c1ff2246dee8e27a626fb8f226 | 0.0149 | rrf |
| 8 | 1cc1381b807247192a4b5103ad26441f | 0.0147 | rrf |
| 9 | 5fd66e935309d642e0f24a46c80f9215 | 0.0145 | rrf |
| 10 | df3b71b91e5b1a485d8f698a7139c2e2 | 0.0143 | rrf |

### 4
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 7428c3a222246edb04bb5ea1d55174e6 | 0.0164 | rrf |
| 2 | 7069db8a1dfae2849dad3dbb37de82ad | 0.0161 | rrf |
| 3 | b9e1576fce7b876a525b4c367c39a574 | 0.0159 | rrf |
| 4 | e4912572cd49fd92d28169f50ff66559 | 0.0156 | rrf |
| 5 | a2468d8bb391f88a32170ea2b3610686 | 0.0154 | rrf |
| 6 | f0ee8063ae1f9abed7dbd81ba3a20bb8 | 0.0152 | rrf |
| 7 | 450f504dff15fb06ad88e8c1b6aa7277 | 0.0149 | rrf |
| 8 | 779bc7aad231f91094c1fc485a9318ab | 0.0147 | rrf |
| 9 | 204f0651e50319b0734b2320f56d6dee | 0.0145 | rrf |
| 10 | 3dca6f296e1c2c504d83970f82d475b0 | 0.0143 | rrf |

### 5
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 0336380bfc3208ef3f8fc21214c976d0 | 0.0164 | rrf |
| 2 | 475645c982fd8cfb30df81ce4923a575 | 0.0161 | rrf |
| 3 | 39e9ab838db21ea75122824583c0d63a | 0.0159 | rrf |
| 4 | 0b89d16e46485edaac905c97d62506d6 | 0.0156 | rrf |
| 5 | c25678121919a201b69dafa8fd0baecd | 0.0154 | rrf |
| 6 | c1f0ea98eab65c8cbb2892489f63f34b | 0.0152 | rrf |
| 7 | 0218fcff041c3345dfe71e8e660324a8 | 0.0149 | rrf |
| 8 | d67035f2410784f888a24b69de61c12e | 0.0147 | rrf |
| 9 | 643587ba0f83f69df6631511daf8f633 | 0.0145 | rrf |
| 10 | a0efb803039940302efd23c9073ae98d | 0.0143 | rrf |

### 9
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 213fbfd5cc4636f502eba8c9d932592b | 0.0164 | rrf |
| 2 | 23837d130ae661668889f6557e02675a | 0.0161 | rrf |
| 3 | 3586c894f8c733514d21fba68497d242 | 0.0159 | rrf |
| 4 | d3a5aab25e15e4e7d9ac9c4042192414 | 0.0156 | rrf |
| 5 | 03187956b026e0a8dc7cf1d55ebbf048 | 0.0154 | rrf |
| 6 | 364a1615e3a8953270d1657e779125fc | 0.0152 | rrf |
| 7 | 9269a89835604232ab9e30202f26e4c6 | 0.0149 | rrf |
| 8 | 6815de6799d48ad606d9401386e00867 | 0.0147 | rrf |
| 9 | f740b87d80b24d89468d1ec87636f00d | 0.0145 | rrf |
| 10 | b51710597ea89d295cc65cf0a68330e7 | 0.0143 | rrf |

### 08
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | e74db9ec91207bce636bd743e34efd48 | 0.0164 | rrf |
| 2 | 83375bad0e822756219d35730677b4a9 | 0.0161 | rrf |
| 3 | 4ec8521cc43922cb958a3de6e8fb8ea1 | 0.0159 | rrf |
| 4 | c1ad4d66d94fc385ede13fdd9557c481 | 0.0156 | rrf |
| 5 | 4bcfe8a1f07712f8f02d980b49090b3c | 0.0154 | rrf |
| 6 | 28ce0153b7c36078875a81ed2109d69d | 0.0152 | rrf |
| 7 | a5254a98033ea5f2d39a94f35e3e55c9 | 0.0149 | rrf |
| 8 | 079b82c65a6a3629a1e8822ac28b287e | 0.0147 | rrf |
| 9 | d24d7d331974e548d6da3828835b5bfb | 0.0145 | rrf |
| 10 | f386c6608512f7bef3aa62eec1483036 | 0.0143 | rrf |

### 8
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 101ef059680feb7b1b8019820dca279b | 0.0164 | rrf |
| 2 | 7ecc5c0e574669bbec7f96e88f075c96 | 0.0161 | rrf |
| 3 | a9e2d0f77dc6865c8b8b8982f551cadb | 0.0159 | rrf |
| 4 | 1c74975dc3887de4d28fd0a628c23566 | 0.0156 | rrf |
| 5 | da09e8d9a316c98f4c89a12536341e4c | 0.0154 | rrf |
| 6 | 5287ba1422ffb0c0ab71705f929d9d56 | 0.0152 | rrf |
| 7 | 515896d282a6db8602822e9da2998516 | 0.0149 | rrf |
| 8 | 1d45af3da520e7f99595833237faaac6 | 0.0147 | rrf |
| 9 | c48b19daabb55eb9a7219b4b0aca7c50 | 0.0145 | rrf |
| 10 | 570cbc77b9aeba4479fb0ab80a23cc92 | 0.0143 | rrf |

### 11
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 8bae49c26b2efbc9891a7b092c895117 | 0.0164 | rrf |
| 2 | 1becc9e9e545aa46a0001a5639d9adc4 | 0.0161 | rrf |
| 3 | 94a4e5c8e0b81a888e140772f1fa0f82 | 0.0159 | rrf |
| 4 | b10f31a24ba6e6145f222a37a5d8a003 | 0.0156 | rrf |
| 5 | c13280256c4cdc22fd7d30c93dd5118d | 0.0154 | rrf |
| 6 | 8e221df89891ee2ba23faf6bbc67b24c | 0.0152 | rrf |
| 7 | 56fc4a0ee9719a0e11e7f5898d432f5e | 0.0149 | rrf |
| 8 | 7c9847ac845a4fb6e21888dd08b63a77 | 0.0147 | rrf |
| 9 | 0b3c87cfd5595a228367bbcc1642c699 | 0.0145 | rrf |
| 10 | 675c4c5b9786e372f4a23055176726d2 | 0.0143 | rrf |

### 09
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | fb9ec9a5cb9544681fc47047eb68b575 | 0.0164 | rrf |
| 2 | a94c68c340fe29079564d4d54043818c | 0.0161 | rrf |
| 3 | 3999f49aab3deb4d156e64abe2941d9d | 0.0159 | rrf |
| 4 | bbb3b7ac8269bd5820af8a3b8e1bb519 | 0.0156 | rrf |
| 5 | 58ef5a1edfe4b63d5d6e22456e6b5b69 | 0.0154 | rrf |
| 6 | 52514809ee0eec1a132ea9e8af945091 | 0.0152 | rrf |
| 7 | 77e4f9b4ba53f1ab7aab57babb092f03 | 0.0149 | rrf |
| 8 | 5cb7a8a921e70ddc01eae925f8dcf1bd | 0.0147 | rrf |
| 9 | bc1a072b14e5be7bce897483cd4be66b | 0.0145 | rrf |
| 10 | 094129c1bcbfb8c6947d1183f74ea6c4 | 0.0143 | rrf |

### 40
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | b87dc6b162b534bbc841e09bbf357967 | 0.0164 | rrf |
| 2 | 4cf567a961d775cf40f7693c1d5dda61 | 0.0161 | rrf |
| 3 | a83f826f7cf465bda37c784a74dd696a | 0.0159 | rrf |
| 4 | ceace7a2f85297db299b4c77a856f05e | 0.0156 | rrf |
| 5 | 1743cb247c60bded7b395e811196c98b | 0.0154 | rrf |
| 6 | b05be14c7d699d7f00054977bbda7734 | 0.0152 | rrf |
| 7 | a6f39a710e192a92ad55c1e3593dd6ed | 0.0149 | rrf |
| 8 | aad7633dff470e4de5f99dbedca27ba4 | 0.0147 | rrf |
| 9 | de08ce4285a14f0a16c8dbe7b1896826 | 0.0145 | rrf |
| 10 | c8465a6eba25aaded37adbe034108d2b | 0.0143 | rrf |

### 10
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 976d356edcd1f3efc82b0017258c28b5 | 0.0164 | rrf |
| 2 | 91388374ae46b5ff2422a05c66fbc616 | 0.0161 | rrf |
| 3 | 80730e89bb8288225025bf03af5b3c3f | 0.0159 | rrf |
| 4 | 9636d82fbf93516347b34f640286b039 | 0.0156 | rrf |
| 5 | d1967c03e53720d56bff325455120820 | 0.0154 | rrf |
| 6 | 17de1d8899c13e8f45272288ae2ebf7f | 0.0152 | rrf |
| 7 | c1146a529460974753a384c8a0590645 | 0.0149 | rrf |
| 8 | 8c17561a83330e5c84ef71a1af17746d | 0.0147 | rrf |
| 9 | b9627cb75db1e4e2939136c2b122b63a | 0.0145 | rrf |
| 10 | 9002f3df29fe2c96c96b8df3bed77018 | 0.0143 | rrf |

### 20
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 760a6874b7da7b87ac8318d91f997321 | 0.0164 | rrf |
| 2 | 1ff3b4b54f177dd536daca0d75657a0c | 0.0161 | rrf |
| 3 | d51ad70fa92235e995d0e6cefa2c6755 | 0.0159 | rrf |
| 4 | bda6e135f33555190109c9a069d339dd | 0.0156 | rrf |
| 5 | 2721d5b13e40570022d69edb3ce858e4 | 0.0154 | rrf |
| 6 | 460ae6154a543ebab9e79422ad22f07e | 0.0152 | rrf |
| 7 | 348ecfcfc2838e5a1fa17862866c0674 | 0.0149 | rrf |
| 8 | b49f2f485cf62272662903ba3adfca18 | 0.0147 | rrf |
| 9 | 096b9679033054e310af0abef840c5ec | 0.0145 | rrf |
| 10 | e945319f84be524510108b8315fe647d | 0.0143 | rrf |

### 13
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 950fa618ad159c8efa53378034f89704 | 0.0164 | rrf |
| 2 | 0d8fb8e478e0e5cf567c1665eca2dc94 | 0.0161 | rrf |
| 3 | 130314d1cf3166e4d7987f37c7d9440b | 0.0159 | rrf |
| 4 | 9f0f77d0f1b4cc8ae196fb07af248ee7 | 0.0156 | rrf |
| 5 | 0662667bed51a16384e5bbf4d3283e3b | 0.0154 | rrf |
| 6 | 262d67794df8287cc8fa72571cabd14a | 0.0152 | rrf |
| 7 | 26d9041d87dd8173e55424f07f3e5265 | 0.0149 | rrf |
| 8 | 48919bb844a4aeb41bd2b3e035584554 | 0.0147 | rrf |
| 9 | f0e2cc5b90a75885d0718d92bf148e98 | 0.0145 | rrf |
| 10 | 0552b55b9665cfd02581837f3460c2c2 | 0.0143 | rrf |

### 62
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 3956f8692035b698d22f3f3b9172f337 | 0.0164 | rrf |
| 2 | 760dd6ee8c9ab13ac9f71b887d026233 | 0.0161 | rrf |
| 3 | f1a52817b49fd64bff17895bc5af0392 | 0.0159 | rrf |
| 4 | 17e31dc247c698aafa227a666020afd0 | 0.0156 | rrf |
| 5 | 61208f0e34a859754b3ad9f38733565c | 0.0154 | rrf |
| 6 | 5525885751a9efe6a6bf335d664c0c3a | 0.0152 | rrf |
| 7 | bb3fb7c386faad3568a5812a33285ce1 | 0.0149 | rrf |
| 8 | 23c6426051bd98a42bca5d1cc148538e | 0.0147 | rrf |
| 9 | 2fa9d9abbe24faa7186b1151364e6e51 | 0.0145 | rrf |
| 10 | b3b0112949be57217c90bab9347fd9e6 | 0.0143 | rrf |

### 65
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 4678e47dce380df4f88696ef937f4d57 | 0.0164 | rrf |
| 2 | 669e190cb8c0b38a6f61ad72039314ec | 0.0161 | rrf |
| 3 | bc3262eaf71c705d8c9e9b998271e990 | 0.0159 | rrf |
| 4 | c579bf0afe8d2d5764f76fd154e5fdec | 0.0156 | rrf |
| 5 | eb2d98fafb573ae5bc71514600518f49 | 0.0154 | rrf |
| 6 | 12921cd58e67ebac1d641247e503594f | 0.0152 | rrf |
| 7 | 3067ceed0aa10b09364bc414edffb2d6 | 0.0149 | rrf |
| 8 | e2c7b914c03ed12d4b49e386f5615674 | 0.0147 | rrf |
| 9 | 2ba094cf2ef8f780503dbc30a5df5546 | 0.0145 | rrf |
| 10 | 2ebb0a225b7fef90348d30dc4d8214ce | 0.0143 | rrf |

### 12
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 46c2a7bdf5b5efc62c28a584f2c804e8 | 0.0164 | rrf |
| 2 | 9b9f1f742ab5fab2db5c5a6d3b2205ed | 0.0161 | rrf |
| 3 | e3ecbc63a4b98bad213a007906723576 | 0.0159 | rrf |
| 4 | dc2e2a85374765da1ece5b7ad2d4a1a2 | 0.0156 | rrf |
| 5 | 545905ac9603b7ae5e18a83203a37e72 | 0.0154 | rrf |
| 6 | 7310b25ed69430bc61ab49ac3c0e08c3 | 0.0152 | rrf |
| 7 | 965ab89bb7c1fe76bfa8207996a9781b | 0.0149 | rrf |
| 8 | 5afd1b55240c38e14cbd1bf57a018c7a | 0.0147 | rrf |
| 9 | 0ed5d3519c8d5d30a973c4341fe91f7a | 0.0145 | rrf |
| 10 | 1cd95f812298252b26e05376382e111d | 0.0143 | rrf |

### 00
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 3e534d5974c274869fa440a7a66896fb | 0.0164 | rrf |
| 2 | b4390380bec5ecb923bd66527c43bfd4 | 0.0161 | rrf |
| 3 | 1eb1357d280cb705bf70ffb848b51894 | 0.0159 | rrf |
| 4 | e9e67131ca8fb64157fc9cadeb2191ea | 0.0156 | rrf |
| 5 | 01aa490a31c26a66cf0a04e3338c0ac3 | 0.0154 | rrf |
| 6 | bac91f82ebfa063badeafa2787351e23 | 0.0152 | rrf |
| 7 | e9c521c913281ac2a8d7cd88e7292e77 | 0.0149 | rrf |
| 8 | a68038248ebf898cc44d85f20e0cafc5 | 0.0147 | rrf |
| 9 | 8ba554c37baa6a7b27b3fe97507257db | 0.0145 | rrf |
| 10 | 1c1b1d5456407d1b8ff2a2de355e582c | 0.0143 | rrf |

### 61
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | a81ce11c020c70b53e0983abb6a58ea2 | 0.0164 | rrf |
| 2 | 7eb9617f536a46475fe17ad5bd9a5064 | 0.0161 | rrf |
| 3 | 598159c00bf5f473592e748802a30349 | 0.0159 | rrf |
| 4 | 0cd439bde400e9a10a0b49d6fcf9de17 | 0.0156 | rrf |
| 5 | 8a5a25b5e0bb9f3753761f55c13db014 | 0.0154 | rrf |
| 6 | e7d1288de241fe06d6a1b8ec40bb19cd | 0.0152 | rrf |
| 7 | a4277b6b4911e60515253a3dcb615e8b | 0.0149 | rrf |
| 8 | 4fe90e074b38f3ab237c80fe6ae77b72 | 0.0147 | rrf |
| 9 | d0f5511672c7369f00d280df8f9501a7 | 0.0145 | rrf |
| 10 | 5d9ea0f56ce16a2c7cee813052ac04cc | 0.0143 | rrf |

### 30
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | e0dbb6c53a8007c03935a108d0b9545e | 0.0164 | rrf |
| 2 | cc6826cd4f613d84a0f2f02af7a545a6 | 0.0161 | rrf |
| 3 | df9e109f8f4c6b772abba46eee727150 | 0.0159 | rrf |
| 4 | 71ca4a3034d78dab6ea554b8e1944f40 | 0.0156 | rrf |
| 5 | 12ff56831d73750606b8c343fc852f3e | 0.0154 | rrf |
| 6 | 00e4309f8142dc9f613662d360294fb5 | 0.0152 | rrf |
| 7 | dbcc18d479c40573f45af6bd5219b29f | 0.0149 | rrf |
| 8 | cc5fd4022f28eac44e67cbb39cded549 | 0.0147 | rrf |
| 9 | e47a1dc99b2d0cca3dfb6c3dad453099 | 0.0145 | rrf |
| 10 | 9bb1d5a26e5b5445f5e62790557d6374 | 0.0143 | rrf |

### 55
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 6a8a0bfcab3d4906cbf2be7a2f31eca6 | 0.0164 | rrf |
| 2 | 8f294bdb48414ab8097e66dab9d442f1 | 0.0161 | rrf |
| 3 | 014c35a3d5fe1a7142d891fbe2aa5209 | 0.0159 | rrf |
| 4 | acaef6a9cabc6c6ef6f20c79424a1aed | 0.0156 | rrf |
| 5 | 70e1fa06fca1c2ff44f81e7060cafd61 | 0.0154 | rrf |
| 6 | 2fa8cda0d1048c84b945e361ff47e3f4 | 0.0152 | rrf |
| 7 | db5c95c7fbba6f277d5b64abd69b133c | 0.0149 | rrf |
| 8 | 837a183e5de7afccf5bda95dfce28051 | 0.0147 | rrf |
| 9 | 9b515bcd464f2ad4d218e682e1894e6c | 0.0145 | rrf |
| 10 | c7048251259481f77c10101af3620d1c | 0.0143 | rrf |

### 21
path: `hybrid`

| # | id | score | leg |
|---|---|---|---|
| 1 | 434e5ba6c1cb0d371e09d693ee81e3b2 | 0.0164 | rrf |
| 2 | 8411a3274103058aa7110ca3ff669f56 | 0.0161 | rrf |
| 3 | ea426c7330cb04885d602cc6c11c8fdd | 0.0159 | rrf |
| 4 | 43410b38063317ab4818d73fc771bc5e | 0.0156 | rrf |
| 5 | 746124cfbee953bc7881ae4da3d8cc8e | 0.0154 | rrf |
| 6 | 5f7b1ad4e06e1d2e8f0f4176be38f0bf | 0.0152 | rrf |
| 7 | 43f3592b9d89703e44a038cbe10df224 | 0.0149 | rrf |
| 8 | fb2559208e6b1645241bfab3f1ce4bb6 | 0.0147 | rrf |
| 9 | cb92d7c8ed2286e2d887e7e58f0ac3b4 | 0.0145 | rrf |
| 10 | 1a599feec93bfbb77d4760c386139436 | 0.0143 | rrf |

## 6. Latency (round-trip from this script through search-api)

| mode | n | p50 ms | p95 ms | mean ms |
|---|---|---|---|---|
| vector | 50 | 55.9 | 60.5 | 54.2 |
| bm25 | 50 | 4.9 | 6.0 | 5.0 |
| hybrid | 50 | 59.8 | 63.1 | 58.0 |
| hybrid_classified | 50 | 55.4 | 63.1 | 56.5 |
